import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10),range(21,31),range(201,211)])
print(x)
#[  0   1   2   3   4   5   6   7   8   9]
#[ 21  22  23  24  25  26  27  28  29  30]
#[201 202 203 204 205 206 207 208 209 210]
print(x.shape) #(3,10)
x=x.T
print(x.shape)#(10,3)
y=np.array([[1,2,3,4,5,6,7,8,9,10]]) #(1,10)
y=y.T
print(y)

#2. 모델 구성

model=Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(8))
model.add(Dense(12))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(20))
model.add(Dense(24))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))    # y의 열의 갯수 만큼 아웃풋 됨

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150, batch_size=1)

#4. 평가, 예측

loss= model.evaluate(x, y)
print("loss :", loss)

result= model.predict([[9, 30, 210]])
print("[[9, 30, 210]]의 예측값 :", result)

#[[9, 30, 210]]의 예측값 : [[9.999885]]
#[[9, 30, 210]]의 예측값 : [[10.000001]]
