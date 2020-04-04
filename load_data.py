#!/usr/bin/env python
# coding: utf-8
import pandas as pd

def load_data(file_name):
    path = './Source/'
    file_extension = '.xlsx' 

    file = path + file_name + file_extension

    df = pd.read_excel(file, index=True)
    
    # print(file_name.find("draw_results"))
    
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
    df[["Draw","1","2","3","4","5","6","Bonus"]] = df[["Draw","1","2","3","4","5","6","Bonus"]].astype(int)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    return df
