import numpy as np
from tensorflow.keras.models import Sequential
from tnesorflow.keras.layers import Dense

#1. data
x=np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [1,1,1,1,2.1.3,1.4,1.5,1.6,1.4],
    [9,8,7,6,5,4,3,2,1,0]
]) #(2,10)
x=x.T
print("x의 행렬은 ",x.shape)#(10,2)

y = np.array([11,12,13,14,15,16,17,18,19,20])

#예측값 [[10,1.4,0]]

#2. 모델 구성
model=Sequential()
model.add(Dense(7,input_dim=3)) # 열이 3개라서 
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss="mse",optimizer='adam')
model.fit(x,y,epochs=30,batch_size=3)

#4. 평가, 예측
loss=model.evaluate(x,y)
print('loss : ',loss)
result=model.predict([[10,1.4,0]])
print('[10,1.4,0]의 예측값은 ', result)

'''
Dense : 3, 7, ,15, 30, 15, 7, 1
optimizer : adam
epochs : 1100
loss(mse) : 1.4051513971935492e-05
[10,1.4,0]의 예측값은 [[19.996786]]
'''
