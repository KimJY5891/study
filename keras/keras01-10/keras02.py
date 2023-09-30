# 1. 데이터
import numpy as np
x = np.array([1,2,3]) # 파이썬 리스트에서 1차원 NumPy 배열을 생성
y = np.array([1,2,3])
#loss는 작을 수록 좋다. 
print(x) # [1 2 3]
print(type(x)) # <class 'numpy.ndarray'>

# 2. 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
# 텐서플로우 안에 케라스 모에 모델스 안에 Sequential를 가지고 와라
from tensorflow.keras.layers import Dense
# 텐서플로우 안에 케라스 안에 레어스 안에 Dense를 가지고 와라
model=Sequential()
model.add(Dense(1,input_dim=1))
# 단일 구성 모델

# 3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=30)
# fit=훈련시키다
# epochs = 훈련양
# 훈련양이 너무 많아도 loss값이 이상하게 될 수 있다. 
# loss가 가장 작은 지점에서 끊어야한다.
# 똑같은 값으로 훈련시켜도 처음에 랜덤하게 시작하기 때문에 다른로결과가 나타날 수 있다.
# loss:0.2880
