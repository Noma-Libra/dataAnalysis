
"""
    
"""
import os
import pandas as pd
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')

userlog_1907 = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'))
# print(userlog_1907.head())

record = pd.DataFrame(userlog_1907, columns=[
                      'idx', 'date', 'hash', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Y']).sort_values(by=['date'])

# print(record.head())
record_list = record.values.tolist()

# 2019-07-01 ~ 31 일 까지의 날짜를 저장하도록 한다. Datetime
xAxis = []


def learning(t_a, t_t, t_r):
    """
        Machine Learning function / Tensorflow2
        t_a -> Target Token(Keyword)
        t_t -> Number of traning
        t_r -> Learning Rate
    """
    import tensorflow as tf
    import numpy as np
