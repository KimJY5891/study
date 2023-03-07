#x의 전체 값을 잘라서 트레인과 테스트 값으로 만들 수 있다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,])
y=np.array([10,9,8,7,6,5,4,3,2,1,])

#[실습] numpy 리스트의 슬라이싱!! 7:3으로 잘라라
x_train= x[:7] #시작 지점인 0은 명시하지 않아도 된다. [:7]
x_test = x[7:] #끝지점은 명시하지 않아도 된다. [7:]
y_train= x[:7] #인덱스 0부터 7미만까지
y_test = x[7:]

print(x_train,x_test) 
print(y_train,y_test)

# 2. 모델구성
model=Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=4)

# 4. 평가, 예측
loss=model.evaluate(x_test,y_test) #트레인값은 없다. 당분간 테스트 값은 훈련에 사용하지 않는다.
print('loss : ',loss)
result=model.predict([[11]])
print('[11]의 예측값은',result)
"""

# 2. 모델구성
model=Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=500,batch_size=4)
loss_train: 4.2443e-12
loss_test :  4.2443084637133754e-12
[11]의 예측값은 [[11.000001]]
"""
