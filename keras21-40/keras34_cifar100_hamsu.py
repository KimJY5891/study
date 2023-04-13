from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
# 마지막 레이어에 노드의 갯수 100개


# 1. 데이터
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print("x shap: ",x_train.shape,x_test.shape)
print("y shape : ",y_train.shape,y_test.shape)
# x shap:  (50000, 32, 32, 3) (10000, 32, 32, 3)
# y shape :  (50000, 1) (10000, 1)


#2. 모델 구성
input01=Input(shape=(32,32,3))
Conv2D_01=Conv2D(10,(2,2),padding='valid',activation='relu')(input01)
MaxPooling2D_01= MaxPooling2D()(Conv2D_01)
Conv2D_02=Conv2D(64,(2,2),padding='valid',activation='relu')(MaxPooling2D_01)
Conv2D_03=Conv2D(32,(2,2))(Conv2D_02)
Flatten_01 = Flatten()(Conv2D_03)
Outputs=Dense(10,activation='softmax')(Flatten_01)


model=Model(inputs=input01,outputs=Outputs)








"""


# 2. 모델 구성
input_01=Input(shape=(28,28,1))
Conv2D_01 =Conv2D(10,(2,2),padding='valid',activation='relu')(input_01)
MaxPooling2D_01=MaxPooling2D()(Conv2D_01)
Conv2D_02=Conv2D(64,(2,2),padding='valid',activation='relu')(MaxPooling2D_01)
Conv2D_03=Conv2D(32,(2,2))(Conv2D_02)
Flatten_01=Flatten()(Conv2D_03)
Outputs=Dense(10,activation='softmax')(Flatten_01)
model= Model(inputs=input_01,outputs=Outputs)


"""
