from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten,
MaxPooling2D
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
# 1. 데이터
(x_train111231sd31fs3d1fs3d1f,y_train),(x_test,y_test) =
mnist.load_data() # 데이터를 넣어줌
scaler = MinMaxScaler()
x_train = x_train.reshape(-1,1)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)
print("x_train.shape01: ",x_train.shape)
print("y_train.shape01 : ",y_train.shape)
#3차원데이터
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1) # 데이터의 구조만 바귀는 것 순서와
값이 바뀌는 게 아님
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)
print(y_train)
print(y_train.shape)
#2. 모델
model =Sequential()
model.add(Conv2D(10,(2,2),padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D()) # conv (cnn)과 다르게 중첩되지 않는다. 계산을 두번
하면 건너 뛰게 되어있다.
model.add(Conv2D(filter=64,kernel_size=(2,2),
padding='valid',
activation='relu'))
model.add(Conv2D(32,2)) #2 = (2,2) 줄임버전
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.summary()
