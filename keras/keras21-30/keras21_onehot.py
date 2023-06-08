# [과제]

# 3가지 원핫인코딩 방식을 비교할 것
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
datasets = load_iris()
x = datasets.data
y = datasets.target
print(y)
# 1. pandas의 get_dummies
#  get_dummies 를 사용하면 type이 dataframe으로 바뀌는데, 다시 numpy로 바꿔줘야한다.

print(type(y))

y = pd.get_dummies(y)
print(y)
print(type(y))

y = np.array(y)
print(y)
print(type(y))
"""
# 2. keras의 to_categorical

print(type(y))
y = to_categorical(y)          # type 그대로
print(type(y))

# label 값으로 0이 생김

# 3. sklearn의 OneHotEncoder
encoder = OneHotEncoder()
y_2d = y.reshape(-1, 1)
print(y_2d)

print(y_2d.shape)
y = encoder.fit_transform(y_2d).toarray()
print(y)

# 미세한 차이를 정리하시오
"""

# scaler = MinMaxScaler()

# scaler = StandardScaler()

# scaler = MaxAbsScaler()

# scaler = RobustScaler()
# 4가지 전부 정리해서 보내기 일요일
