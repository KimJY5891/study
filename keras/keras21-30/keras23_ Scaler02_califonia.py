import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler #전처리
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
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model = Sequential()
model.add(Dense(4,input_dim=8))
model.add(Dense(2))
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

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
hist =model.fit(x_train,y_train,epochs=1000,batch_size=120,
          validation_split=0.2,verbose=1)
print(hist.history) 

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
"""
NON
loss :  0.6167314052581787

scaler = StandardScaler()
loss :  0.5250846147537231

#scaler = MinMaxScaler()
loss :  0.5232634544372559

#scaler = MaxAbsScaler()
loss :  0.5247334837913513

#scaler = RobustScaler()
loss :  0.5263611674308777
"""
# 결과 : MinMaxScaler_승
