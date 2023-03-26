from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from sklearn.metrics import r2_score, accuracy_score
#교육용 데이터셋
from sklearn.datasets import fetch_california_housing
    
#1. 데이터
datasets= fetch_california_housing()
x= datasets.data
y= datasets.target
print("x:",x.shape) #(20640,8)
print("y:",y.shape) #(20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715
)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(4,input_dim=8))
model.add(Dense(2,name="2222"))
model.add(Dense(8))
model.add(Dense(80))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
model.summary()


#함수형 모델 
input01 = Input(shape=(8,)) #input01 = 레이어 이름 
dense01 = Dense(4)(input01)
dense02 = Dense(2)(dense01)
dense03 = Dense(8)(dense02)
dense04 = Dense(80)(dense03)
dense05 = Dense(64)(dense04)
dense06 = Dense(40)(dense05)
dense07 = Dense(32)(dense06)
dense08 = Dense(24)(dense07)
dense09 = Dense(16)(dense08)
dense10 = Dense(12)(dense09)
dense11 = Dense(10)(dense10)
dense12 = Dense(8)(dense11)
dense13 = Dense(4)(dense12)
dense14 = Dense(2)(dense13)
output01 = Dense(1)(dense14)
#함수형 모델 
model = Model(inputs=input01,outputs=output01)
model.summary()
#Model: "model"
#Densedp , name="이름 짓기 가능 "
# 아닐경우 알아서 지음
"""
Model: "sequential"
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 4)                 36
_________________________________________________________________
dense_1 (Dense)  ㅌ          (None, 2)                 10(None, 2)                 10
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 24
_________________________________________________________________
dense_3 (Dense)              (None, 80)                720
_________________________________________________________________
dense_4 (Dense)              (None, 64)                5184
_________________________________________________________________
dense_5 (Dense)              (None, 40)                2600
_________________________________________________________________
dense_6 (Dense)              (None, 32)                1312
_________________________________________________________________
dense_7 (Dense)              (None, 24)                792
_________________________________________________________________
dense_8 (Dense)              (None, 16)                400
_________________________________________________________________
dense_9 (Dense)              (None, 12)                204
_________________________________________________________________
dense_10 (Dense)             (None, 10)                130
_________________________________________________________________
dense_11 (Dense)             (None, 8)                 88
_________________________________________________________________
dense_12 (Dense)             (None, 4)                 36
_________________________________________________________________
dense_13 (Dense)             (None, 2)                 10
_________________________________________________________________
dense_14 (Dense)              (None, 1)                 3
=================================================================
Total params: 11,549
Trainable params: 11,549
Non-trainable params: 0
_________________________________________________________________
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)#시퀀셜과 다르게 INPUT 부터 나온다. 명시하든 아니든 동일한 연산량          [(None, 8)]               0
_________________________________________________________________
dense_15 (Dense)             (None, 4)                 36
_________________________________________________________________
dense_16 (Dense)             (None, 2)                 10
_________________________________________________________________
dense_17 (Dense)             (None, 8)                 24
_________________________________________________________________
dense_18 (Dense)             (None, 80)                720
_________________________________________________________________
dense_19 (Dense)             (None, 64)                5184
_________________________________________________________________
dense_20 (Dense)             (None, 40)                2600
_________________________________________________________________
dense_21 (Dense)             (None, 32)                1312
_________________________________________________________________
dense_22 (Dense)             (None, 24)                792
_________________________________________________________________
dense_23 (Dense)             (None, 16)                400
_________________________________________________________________
dense_24 (Dense)             (None, 12)                204
_________________________________________________________________
dense_25 (Dense)             (None, 10)                130
_________________________________________________________________
dense_26 (Dense)             (None, 8)                 88
_________________________________________________________________
dense_27 (Dense)             (None, 4)                 36
_________________________________________________________________
dense_28 (Dense)             (None, 2)                 10
_________________________________________________________________
dense_29 (Dense)             (None, 1)                 3
=================================================================
Total params: 11,549
Trainable params: 11,549
Non-trainable params: 0
"""
"""

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
hist =model.fit(x_train,y_train,epochs=1000,batch_size=120,
          validation_split=0.2,verbose=1)
print(hist.history) 

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
result =model.predict([17])
print('result : ',result)
"""
"""
NON
loss :  0.6024590730667114
"""
"""
scaler = StandardScaler()
loss :  0.5230822563171387
"""
"""
#scaler = MinMaxScaler()
loss :  0.5251461267471313
"""
"""
#scaler = MaxAbsScaler()
loss :  0.5273013114929199
"""
"""
scaler = RobustScaler()
loss :  0.52208411693573
"""
