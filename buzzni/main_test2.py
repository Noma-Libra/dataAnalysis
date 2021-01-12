import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)
#?

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'log')

userlog = pd.read_csv(os.path.join(DATA_DIR, 'log_1907.csv'), header=0)

def preprocessing_dataset(dataset):
    inputs = {}

    del dataset['idx']
    # 불필요한 idx를 제거
dataset['date'] = pd.to_datetime(dataset['date'].astype(str), format='%Y-%m-%d')
    userlog_features = dataset.copy()
    userlog_labels = userlog_features.pop('Y')
    

    for name, column in userlog_features.items():
        dtype = column.dtype
        if ((dtype == object) or (np.issubdtype(dtype, np.datetime64))):
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1, ), name=name, dtype=dtype)
        # Object 와 datetime64 그리고 숫자 상관 없이 모두 정규화 계층으로 묶는다.
    print(inputs)
    numeric_inputs = {name:input for name, input in inputs.items() if input.dtype == tf.float32}
    # 숫자로만 이루어진 데이터만을 묶는다.
    x = layers.Concatenate()(list(numeric_inputs.values()))
    # ?

    norm = preprocessing.Normalization()
    norm.adapt(np.array(dataset[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    # 숫자로 이루어진 데이터들을 정규화하고 저장

    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
        print(input.dtype)
        # lookup = preprocessing.StringLookup(vocabulary=np.unique(userlog_features[name]))
    #     one_hot =  preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

    #     x = lookup(input)
    #     x = one_hot(x)

    #     preprocessed_inputs.append(x)

    # preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    # titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    # tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)
    

preprocessing_dataset(userlog)