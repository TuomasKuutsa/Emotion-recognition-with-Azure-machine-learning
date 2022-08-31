
from azureml.core.model import Model

import tensorflow as tf

import numpy as np
import os

# ParallelRunStep needs to implement init and run functions.
# Init is the first function to get executed and run is executed by a child node resiving a mini-batch of data.

def init():
    global model
    
    model_path = Model.get_model_path('Emotion-classifier')
    model = tf.keras.models.load_model(model_path)

def run(mini_batch):
    
    inference_results = []

    # process batch
    for file_path in mini_batch:
        
        audio, target = load_audio(file_path)
    
        y_pred = np.argmax(model.predict(audio), axis=-1)[0]
        
        inference_results.append('{}: {}: {}'.format(os.path.basename(file_path), np.argmax(target, axis=0), y_pred))
        
    return inference_results


def load_audio(path):
    
    MAX_LENGTH = 55000
    START_TRIM = 14500
    AUDIO_LENGTH = MAX_LENGTH-START_TRIM

    raw = tf.io.read_file(path)
    audio = tf.audio.decode_wav(raw, desired_channels=1, desired_samples=MAX_LENGTH)[0]
    audio = (audio*(1/tf.math.reduce_max(tf.abs(audio))))[START_TRIM:]
    audio = tf.expand_dims(audio, axis=0)

    target = tf.strings.split(path, sep='-')[-5]
    target = tf.strings.to_number(target, tf.int32)-1
    target = tf.one_hot(target, depth=8)

    return (audio, target)
