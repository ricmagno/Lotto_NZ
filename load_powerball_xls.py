#!/usr/bin/env python
# coding: utf-8
import pandas as pd

def load_xlsx():
    lotto = 'a94c65f6-7123-11ea-835d-1868b10e31b6'
    path = './Source/'
    file_extension = '.xlsx' 

    file = path + lotto + file_extension

    df = pd.read_excel(file, index=True)
    data_lotto = df[6:]

    data_lotto = data_lotto.rename(columns={'Lotto Powerball Winning Number Results': 'Draw',
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

    data_lotto = data_lotto.sort_values(by=['Draw'])
    data_lotto = data_lotto.reset_index(drop=True)
    data_lotto[list("123456")] = data_lotto[list("123456")].astype(int)
    data_lotto['Date'] = pd.to_datetime(data_lotto['Date'], infer_datetime_format=True)
    return data_lotto
