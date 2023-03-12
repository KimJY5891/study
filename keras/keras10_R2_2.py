#강제로 나쁘게 만들자
# 그래야 좋게 만드는 것을 이해할 수 있다. 
"""
조건
1) r2를 음수가 아닌 0.5이하로 만들것
2) 데이터는 건들지 말것
3) 레이어는 인풋푸아웃풋 포함해서 7개 이상
4) batch_size=1  
5) 히든 레이어의 노드는 10개 이상 100개 이하(한레이어당)
6) train_size=0.75
7) epoch 100번이상
8) loss지표는 mse, mae
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sklearn.medel_selection import train_test_split

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,6,17,23,21,20])
x_train, x_test, y_train, y_test = train_test_split(
x,#x_train과 x_test로 분리
y,#y_train과 y_test로 분리
train_size=0.75, shuffle=True, random_state
=2569)

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
