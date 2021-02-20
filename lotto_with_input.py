#!/usr/bin/env python
# coding: utf-8

# https://www.tensorflow.org/tutorials/structured_data/time_series

import sys, getopt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3
from datetime import datetime

from load_data import *
from constants import *


# Ignore warning messages for CPU Advanced Vector Extensions (AVX)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    db = sqlite3.connect("draw_predictions.db")
except:
    print("Please create databse first")

## Input arguments
# ## Parameters

game = 'lotto'
mode_training = True
mode_prediction = False
ver = '002'
univariate = False

univariate_past_history = 80
univariate_future_target = 0

split_percent = .7
BATCH_SIZE = 256
BUFFER_SIZE = 300
EVALUATION_INTERVAL = 300
EPOCHS = 100
ACCURACY_THRESHOLD = .95
LOSS_THRESHOLD = 1.05

past_history = int(input("How many past draws should I consider? "))

# past_history = 1
future_target = 0
STEP = 40

# Add this feature as parameter
sort_balls = False

# For univariate
# balls = ['1', '2', '3', '4', '5', '6', 'Bonus', 'Bonus 2nd', 'Powerball']
balls = ['1', '2', '3', '4', '5', '6', 'Bonus', 'Powerball']

# For Multi-variate
# features_considered = [ '1', '2', '3', '4', '5', '6', 'Bonus', 'Powerball']
features_considered = [ '1', '2', '3', '4', '5', '6']

# print('ARGV      :', sys.argv[1:])
options, remainder = getopt.getopt(sys.argv[1:], 'e:p:s:t:u:v', [
    'univariate=',
    'sort_balls=',
    'mode_training=',
    'mode_prediction=',
    'epochs=',
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
    elif opt in ('-e', '--epochs'):
        EPOCHS = int(arg)
    elif opt in ('-v', '--version'):
        ver = str(arg)

print('\n\n============================')
print('UNIVARIATE      (-u):', univariate)
print('SORTED DATA     (-s):', sort_balls)
print('TRAINING MODE   (-t):', mode_training)
print('PREDICTION MODE (-p):', mode_prediction)
print('EPOCHS          (-e):', EPOCHS)
print('VERSION         (-v):', ver)
print('============================\n\n')

# ## Model naming
sorted_data = 'unsorted'
if sort_balls:
    sorted_data = 'sorted'

model_variables_name = 'multivariate'
if univariate:
    model_variables_name = 'univariate'

simple_lstm_model = []

# ## Functions

class cp_autostop(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("\n OVERFIT %2.2f%%" %(logs.get('val_loss')/logs.get('loss')))
        if(logs.get('val_loss')/logs.get('loss') > LOSS_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(LOSS_THRESHOLD*100))   
            self.model.stop_training = True
            

def checkpoint_path_fun(game, Ball_to_predict, sorted_data, model_variables_name, ver):
    return './Models/' + game + '_Ball_'+ Ball_to_predict + '_' + sorted_data + '_' + model_variables_name + '_' +'V' + str(ver + '/')

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
        indices = range(i-history_size, i)
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
    dft = dft.astype(int)
    return dft

def norm_minmax(data):
    return (data - data.min()) / (data.max() - data.min()), data.min(), data.max()

def scale_minmax(data, _min=1, _max=40):
    return data * (_max - _min) + _min
 
def create_model(out_space,input_def):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(out_space, input_shape=input_def[1:]),
        tf.keras.layers.Dense(40),
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
df = load_data(file_name)

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
            uni_data = uni_data.values
            TRAIN_SPLIT = int(len(dft.index)*split_percent)

            tf.random.set_seed(13)

            [uni_data, data_min, data_max] = norm_minmax(uni_data)
         
            x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                                       univariate_past_history,
                                                       univariate_future_target)

            x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                                   univariate_past_history,
                                                   univariate_future_target)

            train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
            train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

            val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
            val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

            # simple_lstm_model = create_model(past_history, x_train_uni.shape[-2:])
            # dt_string = now.strftime("%Y-%m-%d")
            
            checkpoint_path = checkpoint_path_fun(game, Ball_to_predict, sorted_data, model_variables_name, ver)
            # FIX: LOAD MODEL Checkpoint
            # try: 
                # simple_lstm_model = load_model(checkpoint_path)
                # print('Loading mode checkpoint')
            # except:
                # simple_lstm_model = create_model(past_history, x_train_uni.shape)
            simple_lstm_model = create_model(past_history, x_train_uni.shape)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+'checkpoint',
                                                             save_weights_only=True,
                                                             verbose=1)
            cp_logger = tf.keras.callbacks.TensorBoard(log_dir='Logs',
                                                    write_graph=True,
                                                    histogram_freq=5)

            # print("\n")
            # print("\n Opening tensorboard")
            # print("\n  - on terminal type:")
            # print("\n   `tensorboard --logdir=Logs`")
            # print("\n  - it will returm something like:")
            # print("\n   `TensorBoard 1.14.0a20190603 at http://localhost:6006/ (Press CTRL+C to quit)`")
            # print("\n  - open the browser that that url.")
            # os.system("tensorboard --logdir=Logs")
            # os.system("open http://localhost:6006")

            history_model = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                                  steps_per_epoch=EVALUATION_INTERVAL,
                                  validation_data=val_univariate, validation_steps=50,
                                  callbacks=[cp_callback, cp_logger, cp_autostop()])

            score = simple_lstm_model.evaluate(x_val_uni, y_val_uni, verbose=1)
            save_model(simple_lstm_model, checkpoint_path)


        ## Prediction
        if mode_prediction:
            checkpoint_path = checkpoint_path_fun(game, Ball_to_predict, sorted_data, model_variables_name, ver)
            
            try:
                simple_lstm_model = load_model(checkpoint_path)
            except:
                print('Model could not be loaded. Exiting.')
                exit()

            if past_history == 1:
                print("Format",univariate_past_history)
                # last_result = np.array(input("Enter the last_result: "))
            else:
                print("Format",univariate_past_history)
                last_result = np.reshape(
                np.array(dft[Ball_to_predict].tail(univariate_past_history)),(univariate_past_history,1))
            print("THESE ARE THE PAST RESULTS", last_result)

            [last_result, _min, _max]  = norm_minmax(last_result) 
            last_result = tf.convert_to_tensor([last_result], dtype=np.float64, dtype_hint=None, name=None)
            prediction.append(scale_minmax(simple_lstm_model.predict(last_result), _min, _max))
            print('Prediction of ball %s: %2.2f' %(Ball_to_predict, prediction[-1]))

    if not univariate:

        if Ball_to_predict =='Powerball':
            features_considered = [ '1', '2', '3', '4', '5', '6', 'Bonus', 'Powerball']
        elif Ball_to_predict =='Bonus':
            features_considered = [ '1', '2', '3', '4', '5', '6', 'Bonus']

        dft = clean_my_balls(df, features_considered)
        dataset = dft.values

        y = dft[Ball_to_predict].values

        TRAIN_SPLIT = int(len(dft.index)*split_percent)

        [dataset,_min,_max] = norm_minmax(dataset)
        [y,_min,_max] = norm_minmax(y)

        tf.random.set_seed(13)

        x_train_single, y_train_single = multivariate_data(dataset, y,
                                                           0, TRAIN_SPLIT,
                                                           past_history, future_target,
                                                           STEP, single_step=STEP)

        x_val_single, y_val_single = multivariate_data(dataset, y,
                                                       TRAIN_SPLIT, None,
                                                       past_history, future_target,
                                                       STEP, single_step=True)

        train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
        train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
        val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

        if mode_training:
            checkpoint_path = checkpoint_path_fun(game, Ball_to_predict, sorted_data, model_variables_name, ver)

            single_step_model = create_model(32, x_train_single.shape)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+'checkpoint',
                                                             save_weights_only=True,
                                                             verbose=1)
            cp_logger = tf.keras.callbacks.TensorBoard(log_dir='Logs',
                                                    write_graph=True,
                                                    histogram_freq=5)

            single_step_history = single_step_model.fit(train_data_single,
                                                        epochs=EPOCHS,
                                                        steps_per_epoch=EVALUATION_INTERVAL,
                                                        validation_data=val_data_single,
                                                        validation_steps=50,
                                                        callbacks=[cp_callback, cp_logger, cp_autostop()])
            
            score = single_step_model.evaluate(x_val_single, y_val_single, verbose=1)
            single_step_model.save(checkpoint_path)

    # ## Multivariate prediction
        if mode_prediction:
            checkpoint_path = checkpoint_path_fun(game, Ball_to_predict, sorted_data, model_variables_name, ver)

            try:
                single_step_model = load_model(checkpoint_path)
            except:
                print('Loading unsuccessfully')
                exit()

            last_result = np.array(dft[features_considered].tail(past_history))
            [last_result, _min, _max]  = norm_minmax(last_result)        
            last_result = tf.convert_to_tensor([last_result], dtype=np.float64, dtype_hint=None, name=None)
            if Ball_to_predict == 'Powerball':
                _min = 1
                _max = 2
            print(last_result)
            prediction.append(scale_minmax(single_step_model.predict(last_result), _min, _max))
            print('Prediction of ball %s: %2.2f' %(Ball_to_predict, prediction[-1]))

if mode_prediction:
    print(prediction)
    print('\n=================================================',
          '\n\tPrediction of draw ', df['Draw'].max()+1, ':')
    print('=================================================')
    print('\tUNIVARIATE  :', univariate)
    print('\tSORTED DATA :', sort_balls)
    print('\tVERSION     :', ver)
    print('=================================================')

    j = 0
    for i in balls:
        print('\t\t 🎱  Ball ', i, ': ', int(round(float(prediction[j]),0)), '[', round(float(prediction[j]),2), ']')
        j = j+1
    print('\n=================================================')

    c = db.cursor()
    
    c.execute("insert into lotto values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              [int(df['Draw'].max()+1),
               round(float(prediction[0]),2),
               round(float(prediction[1]),2),
               round(float(prediction[2]),2),
               round(float(prediction[3]),2),
               round(float(prediction[4]),2),
               round(float(prediction[5]),2),
               round(float(prediction[6]),2),
               round(float(prediction[7]),2),
               sort_balls,
               univariate,
               ver])
    db.commit()

db.close()    
exit()
