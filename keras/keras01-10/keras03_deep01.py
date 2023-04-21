# 1.데이터
import numpy as np
#사람이 생각하듯 계산한다.
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델
import tensorflow as tf
from tensorflow.keras.models import Sequential
# 신경망이 밑으로 순차적으로 연산(Sequential)
from tensorflow.keras.layers import Dense

model=Sequential()
# 우리는 Sequential 모델을 사용할 거야
#Sequential를 모델이라 정의할 것이다.
model.add(Dense(3,input_dim=1))
# input_dim = 최초의 인풋 1개
model.add(Dense(7))
model.add(Dense(12))
model.add(Dense(18))
# 중앙은 어떻게 될지 몰라서 hidden 레이어
# 성능은 레이어 값을 보는게 아니라 결과를 보고 판단하는 것이다.
model.add(Dense(22))
model.add(Dense(25))
model.add(Dense(18))
model.add(Dense(10))
model.add(Dense(1))
# output layer
# 안에 내용이나 수치를 바꾸는 것을 튜닝한다고 한다.

#3. 컴파일, 훈련
# 컴파일이란? 기계어로 바꾼다.
# 기계야 내가 한 소스를가 네가 알아들어라~
model.compile(loss='mse',optimizer='adam')
# 평균제곱오차(mse)는 n개의 데이터에오대해 오차의 제곱의 평균으로 정의합니다.
# 제곱값으로 나눠도제괜찮은 것이 어차피 로스 값은 상대적 비교이기 때에에 상관없다.
# optimizer = 최적화
# optimizer = adam은 평타, 한 85%정도 성능나옴
model.fit(x,y,epochs=100)
# loss: 2.8691e-07

