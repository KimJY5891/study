from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#.python  있으면 작동이 안될까? 
import numpy as np

#1. 데이터
x_train =np.array(range(1,17)) #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
y_train =np.array(range(1,17))

x_val = x_train[13:] #x_val : [14 15 16]
y_val = y_train[13:] #y_val : [14 15 16]
x_test = y_train[10:13] #x_test : [11 12 13]
y_test = y_train[10:13] #y_test : [11 12 13]
print('x_test : ',x_train)
print('y_test : ',y_test)

#2. 모델
model=Sequential()
model.add(Dense(2,input_dim=1))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=100,
        validation_data=[x_val,y_val])

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
result =model.predict([17])
print('17의 예측값 : ',result)
"""
loss :  2.7284841053187847e-12
17의 예측값 :  [[17.000004]]
"""
