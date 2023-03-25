import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x=np.array([1,2,3])
y=np.array([1,2,3])

#2.모델 구성
model= Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.summary()
#3. 컴파일, 훈련 

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10()
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 24
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 15(12(웨이트?)+3(바이어스))
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 60
Trainable params: 60
Non-trainable params: 0
_______________________________________________________________

# Output Shape  (None, 5)   (행,열)
# none 상관없다. 
# 바이어스를 연산하기 위해서 
# 웨이트 연산하는데 붙어있다. 
# 써머리 
# 전원을 껐다켜야 메모리가 복구됌
