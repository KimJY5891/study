from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#.python  있으면 작동이 안될까? 
import numpy as np

#1. 데이터
x_train =np.array(range(1,11))
y_train =np.array(range(1,11))
print(x_train)
print(y_train)
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])
print(x_val)
print(y_val)
# 훈련하고 검증
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])

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
