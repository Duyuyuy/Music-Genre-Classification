import json
import os
import math
import librosa
import numpy as np
import ijson
# from tensorflow i
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf

def build_model():

    mel_spec = keras.Input(shape=(128, 640, 1))

    #Define first branch
    x = keras.layers.Conv2D(32, (3, 3),  activation='relu')(mel_spec)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 3), padding='same')(x)
    x1 = keras.layers.AveragePooling2D((24, 213), padding='same')(x)
    x1 = keras.layers.Flatten()(x1)


    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.12)(x)
    x = keras.layers.MaxPooling2D((2, 3), padding='same')(x)
    x2 = keras.layers.AveragePooling2D((32, 32), padding='same')(x)
    x2 = keras.layers.Flatten()(x2)

    x = keras.layers.Conv2D(48, (3, 3), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.12)(x)
    x = keras.layers.MaxPooling2D((2, 3), padding='same')(x)
    # x3 = keras.layers.GlobalAvgPool2D()(x)

    x = keras.layers.Conv2D(48, (3, 3), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.12)(x)
    x = keras.layers.MaxPooling2D((3, 4), padding='same')(x)
    x4 = keras.layers.GlobalAvgPool2D()(x)


    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((4, 4), padding='same')(x)
    x = keras.layers.Dropout(0.15)(x)
    x5 = keras.layers.Flatten()(x)


    #Define second branch
    lstm = keras.layers.Conv2D(32, (3, 3), activation='relu')(mel_spec)
    lstm = keras.layers.BatchNormalization()(lstm)
    lstm = keras.layers.MaxPooling2D((2, 2), padding='same')(lstm)

    lstm = keras.layers.Conv2D(48, (3, 3),  activation='relu')(lstm)
    lstm = keras.layers.BatchNormalization()(lstm)
    lstm = keras.layers.MaxPooling2D((3, 3), padding='same')(lstm)

    lstm = keras.layers.Reshape((106,1008))(lstm)
    lstm = keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))(lstm)
    lstm = keras.layers.LSTM(256, return_sequences= True)(lstm)
    lstm = keras.layers.Dropout(0.15)(lstm)
    lstm = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences= True))(lstm)
    lstm = keras.layers.Dropout(0.15)(lstm)
    lstm = keras.layers.LSTM(128)(lstm)
    lstm = keras.layers.Dropout(0.12)(lstm)
    lstm = keras.layers.Flatten()(lstm)



    concat = keras.layers.Concatenate()(
        [x1,
         x2,
         # x3,
         x4,
         x5,
         lstm
          ])
    outputs = keras.layers.Dense(19, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))(lstm)

    # Define model
    model = keras.models.Model(
        inputs=mel_spec, outputs=outputs)
    return model


model = build_model()

model.summary()
for layer in model.layers:
    print(layer.output)