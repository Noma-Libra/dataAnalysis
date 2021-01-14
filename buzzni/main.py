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

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def preprocessing_dataframe(dataset):
    PREPROCESSING_COLS = ['a','b','c','d','e','f','g','h','Y']
    prep_df = {}
    
    START_DATE = dataset['date'].min()
    END_DATE = dataset['date'].max()
    
    del dataset['idx']
    dataset.set_index('date', inplace=True)

    for date in pd.date_range(start=START_DATE, end=END_DATE).format(formatter=lambda x:x.strftime('%Y-%m-%d')):
        prep_df[date] = dataset.loc[date, PREPROCESSING_COLS].sum()
    
    pref_df = pd.DataFrame(prep_df).transpose()

    return pref_df

def df_to_dataset(dataframe, window_size=5):
    #수정 필요
    FEAUTRES_COLS = ['a','b','c','d','e','f','g','h']
    LABELS_COLS = ['Y']

    START_DATE = dataframe.index.min()
    END_DATE = dataframe.index.max()

    list_feature = np.array(dataframe.loc[START_DATE:END_DATE, FEAUTRES_COLS])
    list_bable = np.array(dataframe.loc[START_DATE:END_DATE, LABELS_COLS])

    return list_feature, list_bable

def dataset_normalization():
    return 0

class TimeWindow():
    split_window = 0
    """
        TimeWindow
    """
    def __init__(self,  train_df, test_df, input_width, label_width, shift, label_columns=None):
        """
            Prams
                @train_df -> 트레이닝 데이터 프레임
                @test_df -> 테스트 테이터 프레임
                @input_width -> 입력 값 갯수, 여기서는 입력된 Timeseries 값 예를들어 1일 경우에는 1일치의 데이터
                @lable_width -> 검출하고자 하는 데이터 갯수, 여기서는 예측할 Timeseries 값 예를들어 2일 경우에는 2일치의 예측 데이터를 도출
                @shift -> 몇일 후 데이터를 뽑을 것인가 지정
                @label_columns -> 예측된 데이터로부터 검증할 값을 지정, 해당 과제는 'Y' 값만을 도출하면 됨
            
            i.e.,
                TEMP = TimeWindow(train_df, test_df, input_width = 5, label_width = 2, shift=7, label_columns=['Y'])
                전처리된 Train, Test 데이터 프레임을 입력 받고, Input_width(=5) 5일간 데이터를 통해서 shift(=7) 7일 뒤인 label_width(=2) 2일간 데이터를 예측할 수 있도록 Window를 생성한다.

        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.train_df = train_df
        self.test_df = test_df
        # 입력값을 클래스의 변수로 저장

        self.total_window_size = input_width + shift
        # total_window_size는 
        
        self.label_columns =label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns) }
            # 도출할 값을 리스트로 변환
        # print(self.label_columns_indices)

        self.column_indices = { name : idx for idx, name in enumerate(train_df.columns) }
        # 학습할 데이터 프레임으로부터 컬럼값들을 저장
        # print(self.column_indices)
        
        self.input_slice = slice(0, input_width)
        # 처음부터 input_width 까지 인덱스를 잘라냄, 즉 몇일 데이터를 입력할 것인가 지정
        
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        # Shift를 포함한 전체 TimeWindowSize에서 학습할 시간열로 지정된 것만큼 잘라내서 저장한다.

        self.label_start = self.total_window_size - self.label_width
        # label_Start 전체 TimeWindowSize에서 도출해내고자 하는 값을 가지고 있는 인덱스에서 부터 시작하도록 지정
        self.labels_slice = slice(self.label_start, None)
        # 도출하고자 하는 시간열에 대한 시작 지점으로부터 끝까지
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        # print(features)
        """
            Params
                @features (batch, time, features)
                    batch
                    time
                    features
            Return
                @inputs (batch, time, features)
                @labels (batch, time, label)
        """
        inputs = features[:, self.input_slice, :]
        # 학습할 데이터들
        labels = features[:, self.labels_slice, :]
        # 검증할 데이터들

        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],axis=-1)
        # 검증할 데이터에 대한 값을 지정

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
        
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )
        print(data)

        ds = ds.map(self.split_window)
        print(ds)
        
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
        
    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

if __name__ == "__main__":
    userlog = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'),header=0, parse_dates=['date'], date_parser=lambda x : pd.to_datetime(x, format='%Y-%m-%d'))
    df = preprocessing_dataframe(userlog)
    # print(df.head())

    column_indices = {name : idx for idx, name in enumerate(df.columns)}
    NUMBER_TIME = len(df)
    # 시계열 데이터의 날짜 수

    NUMBER_FEATURE = df.shape[1]
    # Feature의 갯수 -> [a ,b ,c ,d, ,e, f, g, h]
    # print(NUMBER_FEATURE)

    train_df = df[0:int(NUMBER_TIME*0.7)]
    # 트레이닝 셋 ( 70 % )

    test_df = df[int(NUMBER_TIME*0.7):]
    # 테스트 셋 ( 30 % )

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    
    single_step = TimeWindow(train_df, test_df, input_width = 5, label_width = 5, shift=2, label_columns=['Y'])
    
    """
        i.e.,
            TEMP = TimeWindow(train_df, test_df, input_width = 5, label_width = 2, shift=2, label_columns=['Y'])
            전처리된 Train, Test 데이터 프레임을 입력 받고, Input_width(=5) 5일간 데이터를 통해서 shift(=2) 2일 뒤인 label_width(=2) 2일간 데이터를 예측할 수 있도록 Window를 생성한다.
    """

    # print(np.array(train_df[:w1.total_window_size]))
    # print(np.array(train_df[:w1.total_window_size]).shape)
    """
        ( N, M ) => Train_df에 있는 M개의 컬럼들의 대한 데이터를 N일에 대한 리스트로 나타낸것
        ( 12, 9 ) 
            train_df에 있는 row 12개에 대해서 컬럼 column 값을 저장하고 있는 리스트
    """

    ds = single_step.train
    df = single_step.test

    for x, y in df:
        print(x,y)
    
    

# %%

