from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten ,LSTM, Bidirectional, SimpleRNN, GRU
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터

#2. 모델 
model= Sequential()
#model.add(LSTM(10,2,input_shape=(3,1))) #Total params : 541
model.add(Conv1D(10,2,input_shape=(3,1)))  #Total params : 141
model.add(Conv1D(10,2)) #토탈 파람스 : 301

model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))
model.summary()
'''
model.add(Conv1D(10,2,input_shape=(3,1)))
Conv1D -> 3차원 
속도는 LSTM보다 Conv1D
#3차원 데이터 받아들임 => 특성을 추출하는 역할
#LSTM보다 Total Params양이 더 적기 때문에 Conv1D가 속도 더 빠름 
#Conv2D와 Conv1D 성능 유사, 더 좋을 수도 있음 
'''
