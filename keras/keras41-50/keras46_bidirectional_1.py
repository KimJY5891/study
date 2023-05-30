import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten ,LSTM, Bidirectional, GRU


# 1. 데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
           ,[5,6,7],[6,7,8],[7,8,9]])
y=np.array([4,5,6,7,8,9,10])
print(x.shape,y.shape) #(13, 3) (13,)
x=x.reshape(7,3,1)
print(x.shape,y.shape) #(13, 3) (13,)


# 2. 모델 구성
model=Sequential()
model.add(Bidirectional(LSTM(10,return_sequences=True),input_shape=(3,1)))
model.add(LSTM(10,return_sequences=True))
model.add(Bidirectional(GRU(10,return_sequences=True)))
model.summary()
# ValueError : Please initialize `Bidirectional` layer with a `Layer` instance. You passed: 10
# model.add ( Bidirectional ( SimpleRNN ( 10,input_shape= ( 3 , 1 ) ) ) ) ---> 이런 식으로 안 하고 단독으로 사용하면 오류난다.
'''
Model: "sequential"
SimpleRNN(10) - 120
Bidirectional(SimpleRNN(10) 따블
________________________________________________________________
Layer (type)                 Output Shape              Param #
================================================================
bidirectional (Bidirectional (None, 20)                240
________________________________________________________________
dense (Dense)                (None, 1)                 21
================================================================
Total params: 261
Trainable params: 261
Non-trainable params: 0
_______________________________________________________________
'''
# 앞으로 가는거 하나, 뒤로가는 거 하나 이런식으로 두 번 하는 것이다.
# 바이디렉셔널은 그렇다.
# 트랜스포머 모델
# 자연어 처리 = 나는 밥을 먹었다.
# 바이 디렉셔널이 자연어 처리, 주가예측, 돈 등에서 사용 많이된다.
# LSTM 현역
