#훈련데이터로 평가 하지 않는다.
#객관적인 평가를 하기 위해서는 훈련에 사용한 데이터는 사용하면 안된다.
#우리가 수집한 데이터를 나눈다. 훈련에 사용한 데이터 | 평가용 데이터로 나눈다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,])
y=np.array([10,9,8,7,6,5,4,3,2,1,]) #,옆에 아무것도 적지 않아도 에러가 나지 않는다. 

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# 2. 모델구성
model=Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1100,batch_size=4)

# 4. 평가, 예측
loss=model.evaluate(x_test,y_test) #트레인값은 없다. 당분간 테스트 값은 훈련에 사용하지 않는다.
print('loss : ',loss)
result=model.predict([[11]])
print('[11]의 예측값은',result)
'''
첫 번째 기록 
모델
model=Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))
# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1100,batch_size=4)
loss_train : 1.8190e-12
loss_test :  1.8189894035458565e-12
[11]의 예측값은 [[10.999999]]

'''
