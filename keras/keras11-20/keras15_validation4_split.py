from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x =np.array(range(1,17))
y =np.array(range(1,17))

x=np.array(range(1,17)) #(10,)
y=np.array(range(1,17)) 

x_train=np.array([14,15,16])
y_train=np.array([14,15,16])

x_test=np.array([11,12,13])
y_test=np.array([11,12,13])

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
        validation_split=0.2)

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
result =model.predict([17])
print('17의 예측값 : ',result)
"""
loss :  2.7284841053187847e-12
17의 예측값 :  [[17.000004]]
"""
