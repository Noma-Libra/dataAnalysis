from __future__ import absolute_import, division, print_function, unicode_literals
#%%
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import seaborn as sns
import datetime as dt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')

TEST_SIZE = 20000
# TEST_SIZE : 몇일 간 데이터로 테스트할 것 이냐

WINDOW_SIZE = 5
# WINDOW_SIZE : 얼마만큼의 데이터로 예측할 것이냐

EPOCHS_NUM = 200
# TRAINING 반복 수

"""
분석할 데이터를 전처리
    Parameters
        @dataset ( Type : DataFrame or TextParser) : 읽어오는 전체 데이터 셋
        @keyword ( Type : String ) : dataset에서 검색할 keyword
        @startIndex ( Type : Integer ) : 학습을 시작한 시작일, N - 1 = N 일
        @pastSize : ( Type : Integer ) : 학습할 데이터 양
        @targetSize : ( Type : Integer ) : 예측할 데이터 양

    Return
        @xAxis ( Type : 2nd Array ) :
        @yAxis ( Type : 2nd Array ) :
"""
def showplot_dataset(dataframe, datelable):
    plot_cols = ['a', 'b', 'c','d','e','f','g','h','week','Y']
    plot_features = dataframe[plot_cols]
    plot_features.index = datelable
    _ = plot_features.plot(subplots=True)

    plot_features = dataframe[plot_cols][:20000]
    plot_features.index = datelable[:20000]
    _ = plot_features.plot(subplots=True)

def preprocessing_dataframe(dataset):
    del dataset['idx']
    del dataset['hash']
    # 불필요한 idx,keyword 컬럼 삭제
    

    dataset['week'] = dataset['date'].dt.weekday.map(lambda day: 1 if day >= 4 else 0)

    """
        week 해당 날짜가 주말인지 아닌지에 대한 여부
        0: 월요일 1: 화요일 ... 6: 일요일로 변환
        그리고 date가 주말(금, 토, 일) 은 1의 값 평일은 0으로 지정한다.
    """
    datelabel = dataset.pop('date')

    return dataset, datelabel

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    del dataframe['date']

    dataframe = dataframe.copy()
    labels = dataframe.pop('Y')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


# def makeDataset(data, label, windowSize = 20, keyword=None):
#     feature_list = []
#     label_list = []
#     if keyword is None:
#         print('NONE KEYWORD')
#     else:
#         print(keyword)

#     for idx in range(len(data) - windowSize ):
#         feature_list.append(np.array(data.iloc[idx: idx + windowSize]))
#         label_list.append(np.array(label.iloc[idx + windowSize]))

#     return np.array(feature_list), np.array(label_list)

# def makeModel(feature):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.LSTM(16, 
#                input_shape=(feature.shape[1], feature.shape[2]), 
#                activation='relu', 
#                return_sequences=False)
#           )
#     model.add(tf.keras.layers.Dense(1))

#     return model


# def learn(model, xTrain, yTrain, xValid, yValid):
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#     model_path = 'model'
#     filename = os.path.join(model_path, 'tmp_checkpoint.h5')
#     checkpoint = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

#     history = model.fit(xTrain, yTrain, epochs = EPOCHS_NUM, batch_size=16, validation_data=(xValid, yValid), callbacks=[early_stop, checkpoint])
#     # history = model.fit(xTrain, yTrain, epochs = 200, batch_size=16, validation_data=(xValid, yValid), callbacks=[early_stop, checkpoint])
#     print(history)
#     return history, filename
def main():
    # keyword = '716adcd0325d3422673e820e4cced01f7d84339dc948aafe883f94460cf3de55'
    userlog = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'), header=0, parse_dates=['date'], date_parser=lambda x : pd.to_datetime(x, format='%Y-%m-%d')).sort_values(by='date')


    # CSV파일에 date 컬럼의 값을 datetime으로 변환하고, 날짜순으로 정렬
    print(userlog.describe())

    df = preprocessing_dataframe(userlog)
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    
    print(len(train), '훈련 샘플')
    print(len(val), '검증 샘플')    
    print(len(test), '테스트 샘플')

    feature_columns = []

    for header in ['a','b','c','d','e','f','g','h','week']:
        feature_columns.append(feature_column.numeric_column(header))
    
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    
    model = tf.keras.Sequential([
    feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(train_ds,
            validation_data=val_ds,
            epochs=5)

    loss, accuracy = model.evaluate(test_ds)
    print("정확도", accuracy)
    predY = model.predict(test_ds)
    print(predY)

    feature_cols = ['a','b','c','d','e','f','g','h']
    label_cols = ['Y']

    # train_feature = train[feature_cols]
    # print(train_feature)
    # train_label = train[label_cols]
    # # 트레이닝 셋

    # test_feature = test[feature_cols]
    # test_label = test[label_cols]
    # # 테스트 셋

    # train_feature, train_label = makeDataset(train_feature, train_label, WINDOW_SIZE)
    # xTrain, xValid, yTrain, yValid = train_test_split(train_feature, train_label, test_size = 0.2)
    # # print(xTrain.shape, xValid.shape)

    # test_feature, test_label = makeDataset(test_feature, test_label, WINDOW_SIZE)
    # test_feature.shape, test_label.shape

    # model = makeModel(train_feature)
    # hist, filename = learn(model, xTrain,yTrain, xValid, yValid)

    # model.load_weights(filename)
    # pred = model.predict(test_feature)

    # plt.figure(figsize=(12, 9))
    # plt.plot(test_label, label = 'actual')
    # plt.plot(pred, label = 'prediction')
    # plt.legend()
    # plt.show()
    
if __name__ == "__main__":
    main()

# %%
