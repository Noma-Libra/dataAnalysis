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
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')

KEY_WORD = '716adcd0325d3422673e820e4cced01f7d84339dc948aafe883f94460cf3de55'
# 각 Token(키워드)의 Y(설치 수) 를 예측하는 모델을 만들기 위한 키워드

TRAIN_SPLIT = '2019-07-26'
# 트레이닝 셋으로 이용할 데이터의 양을 지정, 해당 데이터셋은 각 키워드별로 일자를 가지고 있기 때문에 인수를 날짜로 한다.

TRAIN_DATE = pd.to_datetime(TRAIN_SPLIT, format='%Y-%m-%d')
TEST_DATE = TRAIN_DATE + timedelta(days=1)

WINDOW_SIZE = 5
# 몇 WINDOW_SIZE를 통해서 예측할 것인가 지정, 요구사항은 5일에 데이터를 통해서 예측해야하므로 5일인 5를 넣는다.

def showplot_dataset(dataframe, datelable):
    plot_cols = ['a', 'b', 'c','d','e','f','g','h','week','Y']
    plot_features = dataframe[plot_cols]
    plot_features.index = datelable
    _ = plot_features.plot(subplots=True)

    plot_features = dataframe[plot_cols][:20000]
    plot_features.index = datelable[:20000]
    _ = plot_features.plot(subplots=True)

def preprocessing_dataframe(dataset, keyword=KEY_WORD):
    dataframe = dataset.loc[dataset['hash'] == KEY_WORD]
    print(dataframe)
    del dataset['idx']
    norm_cols = ['a','b','c','d','e','f','g','h','Y']
    # 불필요한 idx,keyword 컬럼 삭제
    
    dataset['week'] = dataset['date'].dt.weekday.map(lambda day: True if day >= 4 else False)
    
    """
        week 해당 날짜가 주말인지 아닌지에 대한 여부
        0: 월요일 1: 화요일 ... 6: 일요일로 변환
        그리고 date가 주말(금, 토, 일) 은 True 평일은 False로 지정한다.
    """
    # datelabel = dataset.pop('date')
    """
        plot을 위해서 가져온 데이터의 날짜를 label화
    """
    # for column in dataset.keys():
    #     dtype = dataset[column].dtype
    #     if np.issubdtype(dtype, np.int64):
    #         mean = dataset[column].mean()
    #         std = dataset[column].std()
    #         dataset[column] = (dataset[column] - mean) / std
    
    
    # Boolean 형식인 'week' 컬럼을 제외하곤 데이터를 정규화한다.
    # 데이터 정규화
    # 위 방식은 너무 작은 값에 대해서는 e 값이 등장해 추후 진행되는 코드에 이상을 유발했음...
    # 결측치 제거를 하지 못한게 결함이였다.

    scaler = MinMaxScaler()
    dataset[norm_cols] = scaler.fit_transform(dataset[norm_cols])
    # MinMaxScaler이용

    dataset.set_index('date',inplace=True)
    print(dataset.head())
    return dataset

def df_to_dataset(dataframe, window_size=5):
    features_col = ['a','b','c','d','e','f','g','h']
    labels_col = ['Y']
    np.set_printoptions(precision=8)
    list_feature = []
    list_label = []

    start_date = dataframe.index.min()
    end_date = dataframe.index.max() - timedelta(days=window_size)
    # print(start_date, end_date)
    for date in pd.date_range(start=start_date, end=end_date):
        window = (date + timedelta(days=window_size)).strftime('%Y-%m-%d')
        date = date.strftime('%Y-%m-%d')
        # list_feature.append(np.array(dataframe.loc[date:window, features_col]))
        # list_label.append(np.array(dataframe.loc[date:window, labels_col]))
    # for i in range(len(datafrime) - window_size):
    # list_label.append(dataframe.iloc[1:10])
    # list_label.append(dataframe.iloc[2:11])

    print(dataframe.columns.get_loc('a'))
    # print(np.array(dataframe.iloc[(dataframe.index >= '2019-07-01') & (dataframe.index <= '2019-07-05')]))
    # print(np.array(dataframe.iloc[(dataframe.index >= '2019-07-02') & (dataframe.index <= '2019-07-03')]))
    list_label.append(dataframe.iloc[(dataframe.index >= '2019-07-01') & (dataframe.index <= '2019-07-05')])
    list_label.append(dataframe.iloc[(dataframe.index >= '2019-07-02') & (dataframe.index <= '2019-07-03')])
    # list_feature.append(np.array(dataframe.loc['2019-07-01':'2019-07-05']))
    # list_feature.append(np.array(dataframe.loc['2019-07-02':'2019-07-03']))
    # print(np.array(dataframe.loc['2019-07-01':'2019-07-05', features_col]))
    # print(np.array(dataframe.loc['2019-07-02':'2019-07-03', features_col]))
    # list_label.append(np.array(dataframe['2019-07-01':'2019-07-01']))
    # list_label.append(np.array(dataframe['2019-07-02':'2019-07-02']))
    return np.array(list_feature), np.array(list_label)

if __name__ == "__main__":
    userlog = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'),header=0, parse_dates=['date'], date_parser=lambda x : pd.to_datetime(x, format='%Y-%m-%d')).sort_values(by='date')
    df = preprocessing_dataframe(userlog)
    # showplot_dataset(df, datelabel)
    train = df[:TRAIN_DATE]
    test = df[TEST_DATE:]

    train_feature, train_label = df_to_dataset(train)
    print(train_feature.shape, train_label.shape)

# %%
