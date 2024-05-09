import json
import os
import math
import librosa
import numpy as np
import ijson
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf

def extract_json(filename, key):
    with open(filename, 'rb') as input_file:
         return np.array(list(ijson.items(input_file, key+'.item', use_float=True)))

def prepare(X):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    """
    X = X[..., np.newaxis]
    X_tensor= tf.convert_to_tensor(X)

    return X_tensor


def load_data(JSON_PATH):
    """Loads dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """
    mel_spec = extract_json(JSON_PATH, 'mel_spec')
    y = extract_json(JSON_PATH, 'labels')
    mapping= extract_json(JSON_PATH, 'mapping')
    #
    # mel_spec = mel_spec[..., np.newaxis]


    return mel_spec,y, mapping
