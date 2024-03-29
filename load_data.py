#!/usr/bin/env python
# coding: utf-8
from constants import *
import pandas as pd

def load_data(file_name):
    file = path + file_name + file_extension
    df = pd.read_excel(file, index=False)
    if file_name.find("draw_results") < 5:
        df = df[6:]

    df = df.rename(columns={'Lotto Powerball Winning Number Results': 'Draw',
                                            'Unnamed: 1': 'Date',
                                            'Unnamed: 2': '1',
                                            'Unnamed: 3': '2',
                                            'Unnamed: 4': '3',
                                            'Unnamed: 5': '4',
                                            'Unnamed: 6': '5',
                                            'Unnamed: 7': '6',
                                            'Unnamed: 8': 'Bonus',
                                            'Unnamed: 9': 'Bonus 2nd',
                                            'Unnamed: 10': 'Powerball'})

    df = df.sort_values(by=['Draw'])
    df = df.reset_index(drop=True)
    return data_type(df)

def data_type(df):
    df[["Draw","1","2","3","4","5","6","Bonus"]] = df[["Draw","1","2","3","4","5","6","Bonus"]].astype(int)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    return df

# def ball_result(ball_number):
#     if ball == 7:
#         message = 'Enter Bonus Ball'
#     elif ball == 8:
#         message = 'Enter Second Bonus Ball (0 if not)'
#     elif ball == 9:
#         message = 'Enter PowerBall'
#     else:
#         message = 'Ball number ' + str(ball)
#
#     draw_answer = input("Enter " + message + ":")
#     draw_number = int(draw_answer)
#
#     return draw_number

# while not  :
#     draw_answer = input("Enter Draw number: ")
#     draw_number = int(draw_answer)
#
#     if draw_number != df['Draw'].max() + 1:
#         print('Incorrect Draw number! \nLast Draw was', df['Draw'].max(),
#          '\nExpecting Draw number', df['Draw'].max() + 1)
#     else:
#         correct_draw = True
#         draw_date_answer = input("Enter Draw date (YYYY-MM-DD): ")
#
#     ball_res = []
#
#     for ball in range(1,10):
#         ball_res.append(ball_result(ball))
#
# data_lotto = data_lotto.append(data, ignore_index=True)
# data_lotto.tail()

def update(data):
    df_data = {'Draw':  data[0],
        'Date':      data[1],
        '1':         data[2],
        '2':         data[3],
        '3':         data[4],
        '4':         data[5],
        '5':         data[6],
        '6':         data[7],
        'Bonus':     data[8],
        'Bonus 2nd': 0,
        'Powerball': data[9]}
    df = load_data(file_name)
    draw_number = df['Draw'].max()
    df = df.append(df_data, ignore_index=True)

    # correct_draw = False
    # df.append(last_result)
    # print('Why?')
    # print(df.tail(2), last_result)
    return data_type(df)

def save(df):
    file = path + file_name + file_extension
    df.to_excel(file,index=False)
