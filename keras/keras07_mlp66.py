#x는 3개, y는 3개
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10),range(21,31),range(201,301)])
print(x.shape)#(3,10)
print(x)
x=x.T
print(x.shape)#(10,3)
y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
            [9,8,7,6,5,4,3,2,1,0]
            ])#(3,10)
y=y.T#(3,10)

#실습 예측 : [9,30,210]-[10,1.9,0]
# 2. 모델 구성
model=Sequential()
model.add(Dense(7,input_dim=3)) #열이 3개라서
model.add(Dense(45))
model.add(Dense(90))
model.add(Dense(45))
model.add(Dense(7))
model.add(Dense(3))
#output_dim = y의 열 갯수

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1100)

# 4. 평가, 예측
loss=model.evaluate(x,y)
print('loss : ',loss)
result = model.predict([[9,30,210]])
print('[9,30,210]의 예측값은 ',result)
'''
Dense : 3,7,45,90,45,7,3
epochs=1100
loss(mse):2.3973059648518813e-10
[9,30,210]의 예측값은 [[9.9999466e+00,189996887e+00,1183289e-05]]
'''
