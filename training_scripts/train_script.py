
from azureml.core import Run

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.data import AUTOTUNE

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten,
                                     Conv2D, MaxPooling2D, BatchNormalization,
                                     LayerNormalization, concatenate, Cropping2D)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from kapre.composed import get_melspectrogram_layer
from kapre.signal import LogmelToMFCC

import numpy as np

import argparse, glob, joblib

# ======================================================================
# ==================== Parse script arguments ==========================
# ======================================================================


run = Run.get_context()

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, dest='ds_ref')

# CNN parameters
parser.add_argument('--filters1', type=int, default=80)
parser.add_argument('--filters2', type=int, default=112)
parser.add_argument('--filters3', type=int, default=224)
parser.add_argument('--filters4', type=int, default=256)

parser.add_argument('--kernels1', type=int, default=3)
parser.add_argument('--kernels2', type=int, default=7)
parser.add_argument('--kernels3', type=int, default=5)
parser.add_argument('--kernels4', type=int, default=3)

parser.add_argument('--dense1', type=int, default=192)
parser.add_argument('--dense2', type=int, default=192)

parser.add_argument('--dropout1', type=float, default=0.36)
parser.add_argument('--dropout2', type=float, default=0.50)

# Optimizer parameters
parser.add_argument('--lr', type=float, help='learning_rate', default=0.0001)
parser.add_argument('--bs', type=int, help='batch size', default=16)
parser.add_argument('--epochs', type=int, help='number of epochs', default=120)

args = parser.parse_args()


# ======================================================================
# ==================== Data preparation ================================
# ======================================================================


paths = glob.glob(args.ds_ref + "/*.wav")

train_paths, validation_paths = train_test_split(paths, test_size=0.1,
                                                 stratify=[int(path.split('-')[-5]) for path in paths],
                                                 random_state=11)

MAX_LENGTH = 55000
START_TRIM = 14500
AUDIO_LENGTH = MAX_LENGTH-START_TRIM

def load_audio(path):
    
    raw = tf.io.read_file(path)
    audio = tf.squeeze(tf.audio.decode_wav(raw, desired_channels=1, desired_samples=MAX_LENGTH)[0])
    audio = (audio*(1/tf.math.reduce_max(tf.abs(audio))))[START_TRIM:]
    audio = tf.expand_dims(audio, axis=-1)
    
    target = tf.strings.split(path, sep='-')[-5]
    target = tf.strings.to_number(target, tf.int32)-1
    target = tf.one_hot(target, depth=8)
    
    return (audio, target)


train_data = tf.data.Dataset.from_tensor_slices(train_paths)
train_data = (train_data.shuffle(len(train_paths))
                        .map(load_audio, num_parallel_calls=AUTOTUNE)
                        .batch(args.bs)
                        .prefetch(AUTOTUNE))

val_data = tf.data.Dataset.from_tensor_slices(validation_paths)
val_data = (val_data.map(load_audio, num_parallel_calls=AUTOTUNE)
                    .batch(len(validation_paths))
                    .prefetch(AUTOTUNE))

# ======================================================================
# ==================== Custom layers ===================================
# ======================================================================

@tf.keras.utils.register_keras_serializable()
class RandomGaussianNoise(tf.keras.layers.Layer):
    
    """
    Layer adds gaussian noise to audio as dat augmentation.
    
    prob: probability of applying noise
    stddev: Standard deviation of the added noise. Larger value adds more noise.
    
    """
    
    def __init__(self, prob=0.5, stddev=0.001, **kwargs):
        super(RandomGaussianNoise, self).__init__(**kwargs)
        self.prob = prob
        self.stddev = stddev
    
    @tf.function
    def call(self, inputs, training=None):
        
        if not training:
            return inputs
    
        if tf.less_equal(tf.random.uniform(shape=[]), tf.constant(self.prob, shape=[])):
            return inputs + tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=tf.float32)
        else:
            return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "prob": self.prob,
            'stddev': self.stddev
        })
        return config

@tf.keras.utils.register_keras_serializable()     
class MaskingLayer(tf.keras.layers.Layer):
    
    """
    Creates frequency mask for mel-spectrograms. Standard audio data augmentation technique.
    
    prob: probability of applying a mask.
    n_freq_masks: number of mask strips. Can be overlapping.
    
    """
    
    def __init__(self, prob=0.5, n_freq_masks=1, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)
        self.prob = prob
        self.n_freq_masks = n_freq_masks
     
    def freq_mask(self, spec):
    
        shape = tf.shape(spec)

        time, freq = shape[0], shape[1]

        if tf.less_equal(tf.random.uniform(shape=[]), tf.constant(self.prob, shape=[])):                
            for _ in range(self.n_freq_masks):

                mask_start = tf.random.uniform(shape=[], minval=0, maxval=freq-(freq//4), dtype=tf.int32)
                mask_width = tf.random.uniform(shape=[], minval=2, maxval=8, dtype=tf.int32)

                mask = tf.concat((tf.ones((time, mask_start,1)),
                                  tf.zeros((time, mask_width, 1)),
                                  tf.ones((time, freq-mask_start-mask_width, 1))), axis=1)

                spec = spec*mask + tf.fill(dims=(time, freq, 1), value=-90.)*(1-mask)
            
            return spec
        return spec
        
    def call(self, specs, training=None):
        
        if not training:
            return specs
        
        return tf.map_fn(fn=self.freq_mask, elems=specs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "prob": self.prob,
            "n_freq_masks": self.n_freq_masks,
        })
        return config
    
@tf.keras.utils.register_keras_serializable()   
class MeanVarianceLayer(tf.keras.layers.Layer):
    
    """
    Calculates mean and variance of mel-spectrogram on a given axis. These are used as feature inputs in the model.
    
    Concatenates to Rank 1 feature tensor.
    """

    def __init__(self, axis=1, **kwargs):
        super(MeanVarianceLayer, self).__init__(**kwargs)
        self.axis = axis
    
    def mean_and_var(self, spec):
        
        mean = tf.math.reduce_mean(spec, axis=self.axis)
        var = tf.math.reduce_variance(spec, axis=self.axis)

        return tf.squeeze(tf.concat([mean, var], axis=0), axis=1)
    
    def call(self, specs):
        
        return tf.map_fn(fn=self.mean_and_var, elems=specs, fn_output_signature=tf.float32)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

# ======================================================================
# ==================== Custom Callbacks ================================
# ======================================================================
    
class LogsCallback(keras.callbacks.Callback):
    
    """
    Callback logs performance metrics. 
    """
    
    def __init__(self, validation_data=None):
        super(LogsCallback, self).__init__()
        
        # There is probably a way to get access to validation data without explicitly
        # giving it to the callback since it is all ready given when fit method is called
        self.validation_data = validation_data
    
    def on_train_batch_end(self, batch, logs=None):

        run.log('loss', logs['loss'])
        run.log('accuracy', logs['accuracy'])
        
        
    def on_epoch_end(self, epoch, logs=None):
        
        # Log loss, accuracy and custom f1 score at the end of each epoch
        
        x_val, y_true = next(iter(self.validation_data))      
        y_pred = np.argmax(self.model.predict(x_val), axis=-1)
              
        run.log('val_loss', logs['val_loss'])
        run.log('val_f1', f1_score(np.argmax(y_true, axis=1), y_pred, average='macro'))
        run.log('val_accuracy', logs['val_accuracy'])

        
# ======================================================================
# ==================== Model building ==================================
# ======================================================================


# I chose to build this network such that it takes an audio array as an input without any feature engineering.
# Thus most of the computing is included in the tensorflow computational graph and should benefit from GPU acceleration at least this is the hope.

# I should probably switch to using Pytorch...

def build_model():
    
    input1 = Input((AUDIO_LENGTH,1))
    
    x = RandomGaussianNoise(prob=0.5, stddev=0.002)(input1)
    x = get_melspectrogram_layer(n_fft=512, hop_length=352, return_decibel=True, sample_rate=16000)(x)
    x = MaskingLayer(prob=0.5, n_freq_masks=3)(x)
    
    x2 = MeanVarianceLayer()(x)
    
    x1 = LogmelToMFCC(20)(x)
    x1 = Cropping2D(cropping=((0,0), (1,0)))(x1)
    x1 = LayerNormalization(axis=1)(x1)

    # Convolutional section for MFCCs
    x1 = Conv2D(args.filters1, (args.kernels1, args.kernels1), activation='relu', padding='same', name='Conv1')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(args.filters2, (args.kernels2, args.kernels2), activation='relu', padding='same', name='Conv2')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(3,3))(x1)

    x1 = Conv2D(args.filters3, (args.kernels3, args.kernels3), activation='relu', padding='same', name='Conv3')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(args.filters4, (args.kernels4, args.kernels4), activation='relu', padding='same', name='Conv4')(x1)
    x1 = BatchNormalization()(x1)

    x1 = MaxPooling2D(pool_size=(3,3))(x1)
    
    x1 = Flatten()(x1)
    
    # Normalization and dense layer for Mel-spectrogram mean and variance.
    x2 = BatchNormalization()(x2)
    x2 = Dense(args.dense1, activation='relu')(x2)
    x2 = Dropout(args.dropout1)(x2)
    
    merge = concatenate([x1, x2])
        
    x = Dense(args.dense2, activation='relu')(merge)
    x = Dropout(args.dropout2)(x)
    x = Dense(8, activation='softmax')(x)

    model = Model(inputs=input1, outputs=x)
    
    return model
    
model = build_model()

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=args.lr), metrics=["accuracy"])
    
callbacks = [LogsCallback(val_data),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=0.000025, patience=10, verbose=1),
             EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, verbose=1)]

model.fit(x=train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)

# ======================================================================
# ==================== Model saving ====================================
# ======================================================================

# Save the trained model as an artifact on the outputs folder
os.makedirs('outputs/model', exist_ok=True)

model.save('outputs/model')

print('Model saved!')

run.complete()
