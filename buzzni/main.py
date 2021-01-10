
"""

"""
import os
import pandas as pd
import datetime
import sys


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOG_DIR = os.path.join(BASE_DIR, 'log')

    userlog_1907 = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'))

    # print(userlog_1907.hash.unique())
    # 읽은 CSV파일중 keyword만 읽어옴
    # 716adcd0325d3422673e820e4cced01f7d84339dc948aafe883f94460cf3de55 테스트용 해쉬
    keyword = '716adcd0325d3422673e820e4cced01f7d84339dc948aafe883f94460cf3de55'
    record = pd.DataFrame(userlog_1907[userlog_1907.hash == keyword],
                          columns=[
        'idx', 'date', 'hash', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'Y'
    ]).sort_values(by=['date'])

    record_list = record.values.tolist()
    """
    전처리 끝

    원하는 Keyword 입력시 주어진 CSV파일에서 해당 keyword와 일치하는 값만 날짜순의 Dataframe으로 만든다.
    학습하려는 기간을 입력하기 위해서는 record_list[시작하는 날짜:원하는 날짜] 하면 된다.
    i.e. 5일 학습 record_list[0:5]
    i.e. 6일부터 12일 까지 record_list[5:12]
    """
    # print(record_list[0:4])

    # 입력된 Time series데이터의 데이터타임을 나열
    learning(keyword, record_list, 10, 5000, 0.000003)


def learning(t_a, dataSet, t_d, t_t, t_r):
    """
        Machine Learning function / Tensorflow2
        t_a -> Target Token(Keyword)
        t_t -> Number of traning
        t_r -> Learning Rate
    """

    t_r = (float(t_r))
    t_t = int(t_t)
    import tensorflow as tf
    import numpy as np
    # print(t_a, t_t, t_r)
    # x_train = [sum(data[3:11]) for data in dataSet[0:t_d]]

    x_train = [data[3:11] for data in dataSet[0:t_d]]
    y_train = [[day[11]] for day in dataSet[0:t_d]]

    print(x_train)
    print(y_train)
    # y_train = [sum(data[3:12]) for data in dataSet[0:t_d]]
    # print([sum(data[3:12]) for data in dataSet[0:t_d]])

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(8,)))

    sgd = tf.keras.optimizers.SGD(lr=t_r)
    model.compile(loss='mse', optimizer=sgd)

    # # prints summary of the model to the terminal
    model.summary()
    history = model.fit(np.array(x_train), np.array(y_train), epochs=t_t)

    g_xdata, g_ydata = [], []

    for step in range(t_t):
        if step % 100 == 0:
            cost_val = history.history['loss'][step]
            g_xdata.append(step)
            g_ydata.append(cost_val)
            print("%20i %20.5f" % (step, cost_val)+'\n')

    winner = [82, 3, 80, 242, 0, 0, 0, 0]
    # 64,1,88,88,0,0,0,0 0
    # 60,5,88,440,1,1,0,0 1
    # 91,3,84,253,1,0,0,0 0
    # 82,3,80,242,0,0,0,0 0

    time = model.predict(np.array([winner]))
    print(time)


if __name__ == "__main__":
    # sys.argv[]
    main()
