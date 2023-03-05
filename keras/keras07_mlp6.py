# x는 1개, y는 3개
# 아까와 반대로 x가 하나이고, y가 3개이라면  input_dim과 out_put를 꺼꾸로 작성해주면된다.
# 로스 값이 있기에 훈련이 가능하다.
# 하지만 y가 많은 방식은 실무상으로는 거의 없을 것이다.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x= np.array([range(10)])
print(x.shape) #(1,10)
x=x.T
print(x.shape) #(10,1)
y=np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
    [9,8,7,6,5,4,3,2,1,0]
])
print(y.shape) #(3,10)
y=y.T
print(y.shape) #(10,3)
#실무에서는 데이터가 간겷하지 않다.

#[실습] 예측 : [9] - [10,1.9,0]
# 2. 모델 구성
model = Sequential()
model.add(Dense(7,input_dim=1)) 
model.add(Dense(45))
model.add(Dense(90))
model.add(Dense(45))
model.add(Dense(7))
model.add(Dense(3))
#output_dim = y의 열 갯수

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=200)

# 4. 평가, 에측
loss=model.evaluate(x,y)
print('loss : ',loss)
result=model.predict([[9]])
print('[9]의 에측값은',result)

"""
dense :1, 7,45, 90, 45, 7, 3
loss(mse) : 1.25331043321351281e-08
[9,30,210]의 예측값 : [9.9999771e+00,1.8999461e+00,-3.3842027e-04]
"""
