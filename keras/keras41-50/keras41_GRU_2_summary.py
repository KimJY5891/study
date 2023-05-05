import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,Dropout,LSTM, GRU
#2. 모델 구성
model= Sequential()
#model.add(LSTM(200,input_length=3,input_dim=1))
#model.add(LSTM(20,activation='relu',input_shape=(3,1)))
model.add(GRU(10,activation='relu',input_shape=(5,1)))
#LSTM에서 수정
#
#로직만 다른 것
model.add(Dense(1))
model.summary()
"""
3 * (다음 노드 수^2 + 다음 노드 수 * Shape 의 feature + 다음 노드수 )
3
Layer (type) Output Shape Param #
=================================================================
gru (GRU) (None, 10) 390
dense (Dense) (None, 1) 21
=================================================================
Total params: 1,401
Trainable params: 1,401
Non-trainable params: 0
_________________________________________________________________
업데이트 게이트 와 리셋 게이트의 게이트만
"""
