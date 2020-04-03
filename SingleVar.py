#!/usr/bin/env python
# coding: utf-8

# https://www.tensorflow.org/tutorials/structured_data/time_series

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime




# ## Parameters

mode_training = True
mode_prediction = False

ver = '003'

univariate = True

# Add this feature as parameter
sort_balls = False

# For univariate
balls = ['1', '2', '3', '4', '5', '6', 'Bonus', 'Bonus 2nd', 'Powerball']

# For Multi-variate
features_considered = [ '1', '2', '3', '4', '5', '6']

print('\n\n Running: ',
      '\n\t Univariate', univariate,
      '\n\t Data sorted:', sort_balls,)

# ## Functions

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])

    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))

prediction = []

# ## Data

from load_data import load_data
df = load_data('2020-04-01_draw_results')


# sort_answer = input("Sort balls(True/False)?:")
# sort_balls = bool(sort_answer)

if sort_balls:
    df[['1','2','3','4','5','6']]=df[['1','2','3','4','5','6']].apply(np.sort,axis=1, raw=True, result_type='broadcast')

# Drops the last Draw result for test only
# df.loc[df['Draw'] == 1947]
print('Dropping Draw 1947 for testing. \nREMOVE IT FOR PRODUCTION!!')
df = df[df['Draw'] != 1947]

# ## Ball to Predict
print("Running univariate:", univariate)


if univariate:
    for Ball_to_predict in balls:
        features_considered = [Ball_to_predict]

        dft = df[features_considered]

        for i in features_considered:
            dft = dft.dropna(subset=[i])
            dft = dft[dft[i] != 0]

        ## Training
        if mode_training:
            uni_data = dft
            # print(uni_data.index)
            uni_data.tail()

            uni_data.plot(subplots=True)

            uni_data = uni_data.values

            TRAIN_SPLIT = int(len(dft.index)*.8)
            # print( len(dft.index*.7),TRAIN_SPLIT)

            tf.random.set_seed(13)

            print('Min:',uni_data[:TRAIN_SPLIT].min(),'\nMax:',uni_data[:TRAIN_SPLIT].max())

            uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
            uni_train_std = uni_data[:TRAIN_SPLIT].std()
            uni_data = (uni_data - uni_train_mean) / uni_train_std

            univariate_past_history = 8
            univariate_future_target = 0

            x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                                       univariate_past_history,
                                                       univariate_future_target)

            x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                                   univariate_past_history,
                                                   univariate_future_target)

            # print(x_train_uni.shape, y_train_uni.shape)
            # print(x_val_uni.shape, y_val_uni.shape)

            BATCH_SIZE = 256
            BUFFER_SIZE = 300

            train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
            train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

            val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
            val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

            simple_lstm_model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
                tf.keras.layers.Dense(1)])

            simple_lstm_model.compile(optimizer='adam', loss='mae')

            # for x, y in val_univariate.take(1):
            #     print(simple_lstm_model.predict(x).shape)

            EVALUATION_INTERVAL = 300
            EPOCHS = 100

            simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                                  steps_per_epoch=EVALUATION_INTERVAL,
                                  validation_data=val_univariate, validation_steps=50)

            score = simple_lstm_model.evaluate(x_val_uni, y_val_uni, verbose=1)

            sorted_data = 'unsorted'
            if sort_balls:
                sorted_data = 'sorted'

            # dt_string = now.strftime("%Y-%m-%d")
            simple_lstm_model.save('./Models/' + Ball_to_predict + ver + sorted_data)

        else:
            simple_lstm_model = tf.saved_model.load('./Models/' + Ball_to_predict + ver)

        ## Prediction
        if mode_prediction:

            last_result = np.reshape(
                np.array(dft[Ball_to_predict].tail(8)),(8,1))

            last_result = ((last_result-uni_train_mean)/uni_train_std)

            last_result = tf.convert_to_tensor([last_result,last_result], dtype=np.float64, dtype_hint=None, name=None)

            prediction = [prediction] + [simple_lstm_model.predict(last_result[:1]) * uni_train_std + uni_train_mean]

            print('Prediction of ball', Ball_to_predict, ':', simple_lstm_model.predict(last_result[:1]) * uni_train_std + uni_train_mean)

if mode_prediction:
    print('\n===========================================',
          '\nPrediction:', prediction,
          '\n===========================================')
