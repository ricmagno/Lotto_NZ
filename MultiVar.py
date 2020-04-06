#!/usr/bin/env python
# coding: utf-8

# https://www.tensorflow.org/tutorials/structured_data/time_series

import sys, getopt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Ignore warning messages for CPU Advanced Vector Extensions (AVX)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



## Input arguments
# ## Parameters

mode_training = True
mode_prediction = False
ver = '003'
univariate = False

split_percent = .8
BATCH_SIZE = 256
BUFFER_SIZE = 300
EVALUATION_INTERVAL = 300
EPOCHS = 100



# Add this feature as parameter
sort_balls = False

# For univariate
# balls = ['1', '2', '3', '4', '5', '6', 'Bonus', 'Bonus 2nd', 'Powerball']
balls = ['1', '2', '3', '4', '5', '6', 'Bonus', 'Powerball']

# For Multi-variate
features_considered = [ '1', '2', '3', '4', '5', '6', 'Bonus', 'Powerball']


# print('ARGV      :', sys.argv[1:])
options, remainder = getopt.getopt(sys.argv[1:], 'p:s:t:u:v', [
    'univariate=',
    'sort_balls=',
    'mode_training=',
    'mode_prediction='
    'ver='])

# print ('OPTIONS   :', options)

def str2bool(s):
    answer = []
    if s.lower() in ['true', '1', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
        answer = bool(1)
    elif s.lower() in ['false', '0', 'n', 'no', 'nein', 'nope', 'certainly', 'nah']:
        # answer = not(s.lower() in ['false', '0', 'n', 'no', 'nein', 'nope', 'certainly', 'nah'])
        answer = bool(0)
    else:
        print("\nI don't understand what you mean")
    return answer


for opt, arg in options:
    if opt in ('-u', '--univariate'):
        univariate = str2bool(arg)
    elif opt in ('-s', '--sorted'):
        sort_balls = str2bool(arg)
    elif opt in ('-t', '--training'):
        mode_training = str2bool(arg)
    elif opt in ('-p', '--prediction'):
        mode_prediction = str2bool(arg)
    elif opt in ('-v', '--version'):
        ver = arg

print('\n\n============================')
print('UNIVARIATE      (-u):', univariate)
print('SORTED DATA     (-s):', sort_balls)
print('TRAINING MODE   (-t):', mode_training)
print('PREDICTION MODE (-p):', mode_prediction)
print('VERSION         (-v):', ver)
print('============================\n\n')

# exit()

# ## Model naming
sorted_data = 'unsorted'
if sort_balls:
    sorted_data = 'sorted'

model_variables_name = 'multivariate'
if univariate:
    model_variables_name = 'univariate'

simple_lstm_model = []

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


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))

def clean_my_balls(df, features_considered):
    dft = df[features_considered]
    # if len(features_considered) > 1:
    for i in features_considered:
        dft = dft.dropna(subset=[i])
        dft = dft[dft[i] != 0]
    # else:
        # dft = dft[features_considered].dropna
        # dft = dft[dft[features_considered] != 0]
    return dft

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model

def save_model(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(optimizer=tf.optimizers.Adam(), model=simple_lstm_model)

    print('Saving model in ', checkpoint)
    model.save(checkpoint_path)

    print('Saving weights in ', checkpoint)
    model.save_weights( checkpoint_path.format(epoch=0))
    # model.save_weights( checkpoint_path + '_weights.h5')

    print('Saving model graph_def')
    sess = tf.compat.v1.Session()
    tf.io.write_graph(sess.graph_def, checkpoint_path, 'model.pbtxt')

    return

def load_model(checkpoint_path):
    print('Loading model from ', checkpoint_path)
    model = tf.keras.models.load_model( checkpoint_path )
    print(model)
    model.summary()
    return model

prediction = []

# ## Data

from load_data import load_data
df = load_data('2020-04-01_draw_results')

# sort_answer = input("Sort balls(True/False)?:")
# sort_balls = bool(sort_answer)

if sort_balls:
    df[['1','2','3','4','5','6']]=df[['1','2','3','4','5','6']].apply(np.sort,axis=1, raw=True, result_type='broadcast')

for Ball_to_predict in balls:
    if univariate:
        features_considered = [Ball_to_predict]

        # replaced by clean_my_balls
        dft = clean_my_balls(df, features_considered)

        ## Training
        if mode_training:
            uni_data = dft
            # print(uni_data.index)
            # uni_data.tail()
            # uni_data.plot(subplots=True)
            uni_data = uni_data.values

            TRAIN_SPLIT = int(len(dft.index)*split_percent)

            tf.random.set_seed(13)

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

            train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
            train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

            val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
            val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

            simple_lstm_model = create_model()
            # dt_string = now.strftime("%Y-%m-%d")
            checkpoint_path = './Models/' + 'Ball_'+ Ball_to_predict + '_' + sorted_data + '_' + model_variables_name + '_' +'V' + str(ver + '/')
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+'checkpoint',
                                                             save_weights_only=True,
                                                             verbose=1)
            cp_logger = tf.keras.callbacks.TensorBoard(log_dir='Logs',
                                                    write_graph=True,
                                                    histogram_freq=5)
            
            simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                                  steps_per_epoch=EVALUATION_INTERVAL,
                                  validation_data=val_univariate, validation_steps=50,
                                  callbacks=[cp_callback, cp_logger])
            ## Opening tensorboard
            #  - on terminal type:
            #   `tensorboard --logdir=logs`
            #  - it will returm something like:
            #   `TensorBoard 1.14.0a20190603 at http://what.local:6006/ (Press CTRL+C to quit)`
            #  - open the browser that that url.

            score = simple_lstm_model.evaluate(x_val_uni, y_val_uni, verbose=1)
            save_model(simple_lstm_model, checkpoint_path)
 

        ## Prediction
        if mode_prediction:
            checkpoint_path = './Models/' + 'Ball_'+ Ball_to_predict + '_' + sorted_data + '_' + model_variables_name + '_' +'V' + str(ver + '/')
            try:
                simple_lstm_model.summary()
            except:
                try:
                    simple_lstm_model = load_model(checkpoint_path)
                except:
                    print('Model could not be loaded. Exiting.')
                    exit()

            last_result = np.reshape(
                np.array(dft[Ball_to_predict].tail(8)),(8,1))
            print('Last result', last_result)
            # print('Min:',uni_data[:TRAIN_SPLIT].min(),'\nMax:',uni_data[:TRAIN_SPLIT].max())
            sample_mean = dft[Ball_to_predict].mean()
            sample_std = dft[Ball_to_predict].std()
            last_result = (last_result - sample_mean) / sample_std
            # uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
            # uni_train_std = uni_data[:TRAIN_SPLIT].std()
            # uni_data = (uni_data - uni_train_mean) / uni_train_std
            # last_result = ((last_result-uni_train_mean)/uni_train_std)
            last_result = tf.convert_to_tensor([last_result,last_result], dtype=np.float64, dtype_hint=None, name=None)

            prediction = [prediction] + [simple_lstm_model.predict(last_result[:1]) * sample_std + sample_mean]

            print('Prediction of ball', Ball_to_predict, ':', simple_lstm_model.predict(last_result[:1]) * sample_std + sample_mean)
            exit()

    if not univariate:
        print(df)

        if Ball_to_predict =='Powerball':
            features_considered = [ '1', '2', '3', '4', '5', '6', 'Bonus', 'Powerball']
        elif Ball_to_predict =='Bonus':
            features_considered = [ '1', '2', '3', '4', '5', '6', 'Bonus']
        

        dft = clean_my_balls(df, features_considered)
        dataset = dft.values
        y = dft[Ball_to_predict].values
        
        TRAIN_SPLIT = int(len(dft.index)*split_percent)
        
        sample_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
        sample_std = dataset[:TRAIN_SPLIT].std(axis=0)
        dataset = (dataset - sample_mean) / sample_std
        y = (y - y.mean()) / y.std()
        
        past_history = 20
        future_target = 1
        STEP = 6

        tf.random.set_seed(13)

        x_train_single, y_train_single = multivariate_data(dataset, y, 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)

        x_val_single, y_val_single = multivariate_data(dataset, dataset[:, [1]],
                                                       TRAIN_SPLIT, None, past_history,
                                                       future_target, STEP, single_step=True)

        print ('Single window of past history : {}'.format(x_train_single[0].shape))
        print(x_train_single[-1])

        train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
        train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
        val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

        if mode_training:
            single_step_model = tf.keras.models.Sequential()
            single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
            single_step_model.add(tf.keras.layers.Dense(1))
            single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

            for x, y in val_data_single.take(1):
                print(single_step_model.predict(x).shape)


                single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                            steps_per_epoch=EVALUATION_INTERVAL,
                                                            validation_data=val_data_single,
                                                            validation_steps=50)        

                score = single_step_model.evaluate(x_val_single, y_val_single, verbose=1)
                # dt_string = now.strftime("%Y-%m-%d")
                single_step_model.save('./Models/' + 'Ball_'+ Ball_to_predict + '_' + sorted_data + '_' + model_variables_name + '_' +'V' + str(ver))

    # ## Multivariate prediction
        if mode_prediction:
            if not single_step_model:
                try:
                    single_step_model = tf.saved_model.load('./Models/' + 'Ball_'+ Ball_to_predict + '_' + sorted_data + '_' + model_variables_name + '_' + 'V' + ver)
                except:
                    print('Loading unsuccessfully')
                    exit()

            # last_result = np.reshape(np.array(dft[features_considered].tail(STEP-2)),(8,6))
            last_result = np.array(dft[features_considered].tail(STEP-2))
            print('Last result', last_result)
            sample_mean = dft[Ball_to_predict].mean()
            sample_std = dft[Ball_to_predict].std()
            last_result = (last_result - sample_mean) / sample_std
            last_result = tf.convert_to_tensor([last_result,last_result], dtype=np.float64, dtype_hint=None, name=None)
            prediction = [prediction] + [single_step_model.predict(last_result[:1]) * sample_std + sample_mean]

            print('Prediction of ball', Ball_to_predict, ':', single_step_model.predict(last_result[:1]) * sample_std + sample_mean)


if mode_prediction:
    print('\n=================================================', '\nPrediction of draw ', df['Draw'].max(), ':')
    for i in balls:
        print('\n\t Ball ', i, ': ', prediction)
    print('\n=================================================')
exit()
