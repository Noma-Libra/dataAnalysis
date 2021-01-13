#%%
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

TRAIN_SPLIT = '2019-07-19'
# 트레이닝 셋으로 이용할 데이터의 양을 지정, 해당 데이터셋은 각 키워드별로 일자를 가지고 있기 때문에 인수를 날짜로 한다.

TRAIN_DATE = pd.to_datetime(TRAIN_SPLIT, format='%Y-%m-%d')

TEST_DATE = TRAIN_DATE + timedelta(days=1)

WINDOW_SIZE = 5
# 몇 WINDOW_SIZE를 통해서 예측할 것인가 지정, 요구사항은 5일에 데이터를 통해서 예측해야하므로 5일인 5를 넣는다.

def preprocessing_dataframe(dataset):
    del dataset['idx']
    # 불필요한 컬럼 idx 삭제

    ORIGINAL_COLS = dataset.columns.tolist()
    # ['date', 'hash', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Y']

    START_DATE = dataset['date'].min()
    # 받아온 데이터 셋에서 가장 작은 데이트타임을 가져옴
    END_DATE = dataset['date'].max()
    # 받아온 데이터 셋에서 가작 큰 데이트타임을 가져옴
    
    TIME_INDEX_LIST = pd.date_range(start=START_DATE, end=END_DATE)
    # 받아온 데이터의 가장 큰 데이트타임과 작은 데이트타임 사이에 대한 시간열 리스트

    KEYWORDS_LIST = dataset.hash.unique()
    # 받아온 데이터의 전체 고유한 Hash(Keyworkd)값

    DATE_KEYWORD_IDX = pd.MultiIndex.from_product([TIME_INDEX_LIST, KEYWORDS_LIST])
    prep_dataset = dataset.set_index(['date','hash']).reindex(DATE_KEYWORD_IDX, fill_value=0.0)
    # 각 키워드가 날짜별로 검색량이 다르기 때문에 비워져있는 시간열에 모든 키워드들을 맞춰준다.

    prep_dataset.reset_index(inplace=True)
    prep_dataset.columns = ORIGINAL_COLS

    prep_dataset['week'] = prep_dataset['date'].dt.weekday.map(lambda day: True if day >= 4 else False)
    """
        week 해당 날짜가 주말인지 아닌지에 대한 여부
        0: 월요일 1: 화요일 ... 6: 일요일로 변환
        그리고 date가 주말(금, 토, 일) 은 True 평일은 False로 지정한다.
    """
    NORM_COLS = ['a','b','c','d','e','f','g','h','Y']
    # 정규화할 데이터 목록

    scaler = MinMaxScaler()
    prep_dataset[NORM_COLS] = scaler.fit_transform(prep_dataset[NORM_COLS])
    prep_dataset.set_index('date', inplace=True)
    # MinMaxScaler이용해서 전체 데이터 정규화
    
    # for column in dataset.keys():
    #     dtype = dataset[column].dtype
    #     if np.issubdtype(dtype, np.int64):
    #         mean = dataset[column].mean()
    #         std = dataset[column].std()
    #         dataset[column] = (dataset[column] - mean) / std
    
    # Boolean 형식인 'week' 컬럼을 제외하곤 데이터를 정규화한다.
    # 위 방식은 너무 작은 값에 대해서는 e 값이 등장해 추후 진행되는 코드에 이상을 유발할 경우가 있음

    return prep_dataset

def df_to_dataset(dataframe, window_size=5):
    #수정 필요
    FEAUTRES_COLS = ['a','b','c','d','e','f','g','h']
    LABELS_COLS = dataframe['Y']

    START_DATE = dataframe.index.min()
    END_DATE = dataframe.index.max() - timedelta(days=window_size)

    list_feature = []
    list_label = []

    # print(start_date, end_date)
    for date in pd.date_range(start=START_DATE, end=END_DATE):
        window = (date + timedelta(days=window_size)).strftime('%Y-%m-%d')
        date = date.strftime('%Y-%m-%d')
        print(f'{date} ~ {window}')
        
        list_feature.append(np.array(dataframe.loc[date:window, FEAUTRES_COLS]))
        list_label.append(np.array(dataframe.loc[date:window, 'Y']))

    return np.array(list_feature), np.array(list_label)

def make_model(feature):
    model = Sequential()

    model.add(LSTM(16, 
               input_shape=(feature.shape[1], feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
    model.add(Dense(1))
   
    return model

def learning(model, x_train, y_train, x_valid, y_valid, MODEL_FILE_NAME='TEMP.h5'):
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    filename = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(x_train, y_train, 
                        epochs=200, 
                        batch_size=36,
                        validation_data=(x_valid, y_valid), 
                        callbacks=[early_stop, checkpoint])
    return history, filename

if __name__ == "__main__":
    userlog = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'),header=0, parse_dates=['date'], date_parser=lambda x : pd.to_datetime(x, format='%Y-%m-%d'))
    df = preprocessing_dataframe(userlog)
    # print(df)
    # showplot_dataset(df, datelabel)
    train = df[:TRAIN_DATE]
    test = df[TEST_DATE:]

    train_feature, train_label = df_to_dataset(train)
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

    # print(train_feature)

    test_feature, test_label = df_to_dataset(test)
    # print(test_label)
    

    train_model = make_model(train_feature)
    history, file_name = learning(train_model, x_train, y_train, x_valid, y_valid)
    # model.load_weights(filename)

    # 예측
    pred = train_model.predict(test_feature)
    print(pred)
    print(test_label)
# %%
