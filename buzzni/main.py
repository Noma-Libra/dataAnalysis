from __future__ import absolute_import, division, print_function, unicode_literals
# %%
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')

TEST_SIZE = 10
WINDOW_SIZE = 5
EPOCHS_NUM = 200

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


def preprocessing(dataset, keyword):
    dataset['date'] = pd.to_datetime(
        dataset['date'].astype(str), format='%Y-%m-%d')
    # CSV파일에 date 컬럼의 값을 datetime으로 변환

    dataFrame = dataset.loc[dataset['hash'] ==
                            keyword].sort_values(by='date', ascending=True)
    # 검색하고자 하는 키워드에 해당하는 값만 인덱싱 후 날짜 순으로 정렬
    dataFrame.drop(["idx", "date", "hash"], axis=1, inplace=True)
    # 날짜 순으로 정렬한 뒤 필요없는 데이터 idx, date,hash 를 삭제

    scaler = MinMaxScaler()
    scaler_cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Y']
    dataFrame_scaled = scaler.fit_transform(dataFrame[scaler_cols])
    dataFrame_scaled = pd.DataFrame(dataFrame_scaled)
    dataFrame_scaled.columns = scaler_cols
    # 시계열 데이터들을 스칼라 값으로 정규화
    # 정규화를 왜? 전체 데이터를 0,1사이의 값으로 만들어 학습에 도움을 줌

    return dataFrame_scaled


def makeDataset(data, label, windowSize=20):
    feature_list = []
    label_list = []

    for idx in range(len(data) - windowSize):
        feature_list.append(np.array(data.iloc[idx: idx + windowSize]))
        label_list.append(np.array(label.iloc[idx + windowSize]))

    return np.array(feature_list), np.array(label_list)


def makeModel(feature):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(16,
                                   input_shape=(
                                       feature.shape[1], feature.shape[2]),
                                   activation='relu',
                                   return_sequences=False)
              )
    model.add(tf.keras.layers.Dense(1))

    return model


def learn(model, xTrain, yTrain, xValid, yValid):
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)

    model_path = 'model'
    filename = os.path.join(model_path, 'tmp_checkpoint.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(xTrain, yTrain, epochs=EPOCHS_NUM, batch_size=16, validation_data=(
        xValid, yValid), callbacks=[early_stop, checkpoint])
    # history = model.fit(xTrain, yTrain, epochs = 200, batch_size=16, validation_data=(xValid, yValid), callbacks=[early_stop, checkpoint])
    print(history)
    return history, filename


def main():
    keyword = '716adcd0325d3422673e820e4cced01f7d84339dc948aafe883f94460cf3de55'
    userlog = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'), header=0)

    df = preprocessing(userlog, keyword)
    # 전처리된 데이터를 가져옵니다.

    train = df[:-TEST_SIZE]
    test = df[-TEST_SIZE:]

    feature_cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    label_cols = ['Y']

    train_feature = train[feature_cols]
    train_label = train[label_cols]
    # 트레이닝 셋

    test_feature = test[feature_cols]
    test_label = test[label_cols]
    # 테스트 셋

    train_feature, train_label = makeDataset(
        train_feature, train_label, WINDOW_SIZE)
    xTrain, xValid, yTrain, yValid = train_test_split(
        train_feature, train_label, test_size=0.2)
    # print(xTrain.shape, xValid.shape)

    test_feature, test_label = makeDataset(
        test_feature, test_label, WINDOW_SIZE)
    test_feature.shape, test_label.shape

    model = makeModel(train_feature)
    hist, filename = learn(model, xTrain, yTrain, xValid, yValid)

    model.load_weights(filename)
    pred = model.predict(test_feature)

    plt.figure(figsize=(12, 9))
    plt.plot(test_label, label='actual')
    plt.plot(pred, label='prediction')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

# %%
