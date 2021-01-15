# %%
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM

from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# KEY_WORD = '716adcd0325d3422673e820e4cced01f7d84339dc948aafe883f94460cf3de55'
# 각 Token(키워드)의 Y(설치 수) 를 예측하는 모델을 만들기 위한 키워드

TRAIN_SPLIT = '2019-07-26'
# 트레이닝 셋으로 이용할 데이터의 양을 지정, 해당 데이터셋은 각 키워드별로 일자를 가지고 있기 때문에 인수를 날짜로 한다.

TRAIN_DATE = pd.to_datetime(TRAIN_SPLIT, format='%Y-%m-%d')

TEST_DATE = TRAIN_DATE + timedelta(days=1)

TRAIN_RATE = 0.00003
# 트레이닝 레이트

MODEL_FILE_NAME = 'TEMP.h5'
#

EPOCHS_NUM = 2000

WINDOW_SIZE = 5
# 몇 WINDOW_SIZE를 통해서 예측할 것인가 지정, 요구사항은 5일에 데이터를 통해서 예측해야하므로 5일인 5를 넣는다.

SHIFT = 2
# 예측을 위한 SHIFT


def preprocessing_dataframe(dataset):
    PREPROCESSING_COLS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Y']
    prep_df = {}

    START_DATE = dataset['date'].min()
    END_DATE = dataset['date'].max()

    del dataset['idx']
    dataset.set_index('date', inplace=True)

    for date in pd.date_range(start=START_DATE, end=END_DATE).format(formatter=lambda x: x.strftime('%Y-%m-%d')):
        prep_df[date] = dataset.loc[date, PREPROCESSING_COLS].sum()

    pref_df = pd.DataFrame(prep_df).transpose()

    return pref_df


def df_to_dataset(dataframe):
    # 수정 필요
    labels = dataframe['Y']
    list_feature = []
    list_label = []

    TOTAL_WINDOW = WINDOW_SIZE + SHIFT
    for idx in range(len(dataframe) - TOTAL_WINDOW):
        # 0 ~ 전체 데이터프레임 값 - TOTAL_WINDOW
        list_feature.append(
            np.array(dataframe.iloc[idx:idx+WINDOW_SIZE]))
        # 5일간 데이터들의 리스트를 하나의 원소로 가지는 리스트를 생성한다. (batch, days, col num)
        list_label.append(
            np.array(labels.iloc[idx + WINDOW_SIZE: idx + TOTAL_WINDOW]).reshape(2, 1))

    # 5일 뒤 데이터 이후의 구하려는 값을 원소로 하는 리스트 생성
    return np.array(list_feature), np.array(list_label)
    # return 0


def make_model(feature):
    model = keras.Sequential()
    model.add(LSTM(1,
                   input_shape=(feature.shape[1], feature.shape[2]),
                   activation='relu',
                   return_sequences=False)
              )
    model.summary()
    return model


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def learning(model, x_train, y_train, x_valid, y_valid, MODEL_FILE_NAME='TEMP.h5', patience=2):

    return 0


if __name__ == "__main__":
    userlog = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'), header=0, parse_dates=[
                          'date'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
    df = preprocessing_dataframe(userlog)

    column_indices = {name: i for i, name in enumerate(df.columns)}

    feature, label = df_to_dataset(df)
    train_feature, test_feature, train_label, test_label = train_test_split(
        feature, label, test_size=0.3
    )
    train_feature, valid_feature, train_label, valid_label = train_test_split(
        train_feature, train_label, test_size=0.2
    )
    print(train_feature.shape, train_label.shape)
    print(test_feature.shape, test_label.shape)
    print(valid_feature.shape, valid_label.shape)
    """
        전체 데이터 셋에 80% 를 트레이닝 셋으로 20% 를 테스트 셋으로 Split

    """

    # (16, 5, 9) (16, 1, 1) // (batch, days, input col num) (batch, days, output col num)

    # x_train, x_valid, y_train, y_valid = train_test_split(
    #     train_feature, train_label, test_size=0.3)
    # # print(x_valid, y_valid)
    # print(x_valid.shape, y_valid.shape)

    # test_feature, test_label = df_to_dataset(test)
    # print(test_feature.shape, test_label.shape)
    # # (5, 5, 9) (5, 5, 9)
    # print(test_feature)
    # train_model = make_model(train_feature)

    # history, file_name = learning(train_model, x_train, y_train, x_valid, y_valid)
    # model.load_weights(filename)

    # 예측
    # pred = train_model.predict(test_feature)
    # print(test_feature)

    # plt.figure(figsize=(12, 9))
    # plt.plot(test_label, label='actual')
    # plt.plot(pred, label='prediction')
    # plt.legend()
    # plt.show()
# %%
