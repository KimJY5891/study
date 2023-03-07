#x는 3개
#y는 2개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([range(10),range(21,31),range(201,211)]) #(2,10)
x=x.T
print(x.shape)#(10,3)
y=np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
])#(2,10)
y=y.T

#[실습] 얘측 [9,30,210] - [10,1.9]
# 2. 모델 구성
model = Sequential()
model.add(Dense(7,input_dim=3))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(2))
#output_dim=y의 열 갯수

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1000)

# 4. 평가, 예측
loss=model.evaluate(x,y)
print('loss : ',loss)
result=model.predict([[9,30,210]])
print('[9,30,210]의 예측값은 ',result)
"""
dense :3,7,15,7,2
loss(m) :4.402365583189294e-09
[9,30,210]의 예측값 : [9.999939,1.9001225]
"""
