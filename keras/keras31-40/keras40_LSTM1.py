#LSTM
# rnn의단 점을 보완 
# 너무 길어져서 먼위치에 있는 정보를 기억할 수 없다는 단점을 보완 
# rnn의 개량형 통상적으로 4배의 연산을 더한다. 
# 배니싱 = 사라진다.는 의미
# 처음 스테이스값이 마지막가지 못간다. 
# 하지만 어느정도 지나면 기억하려는 것도 어느 정도 망각된다. 
# rnn 만들어 보라고 하면 LSTM을 만들게 된다. 가장 성능이 좋다. 
# 심플RNN보다 LSTM이 성능이 더좋다. 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,Dropout, LSTM

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
#y=?
#시계열 은 y가 없다. 
x= np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
#8.9.10 넣으면 y 데이터를 만들수 없다. 
y = np.array([6,7,8,9,10])
print(x.shape,y.shape)#(7, 3) (7,)
# x의shape = (행,열,몇개씩 훈련하는지)
# 3차원 데이터라 reshape 해줘야한다.
x=x.reshape(5,5,1)
print(x)#[[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]], [[5],[6] ...
# dnn = 2차원 , cnn = 4차원, rnn=3c차원
print(x.shape) #(5, 5, 1)

#2.모델 구성
model=Sequential()
model.add(LSTM(200,input_shape=(5,1))) #input 모두가 행빼고 나머지 이듯 
model.add(Dropout(0.5))
model.add(Dense(8,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(32))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
import time
start = time.time()
# from tensorflow.python.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss',patience=40,mode='min',verbose=1,restore_best_weights=True)
model.fit(x,y,epochs=1000,batch_size=100)
end = time.time()
#4. 평가 예측
loss =model.evaluate(x,y)
x_predict = np.array([6,7,8,9,10]).reshape(1,5,1) #[[6],[7],[8],[9],[10]]
print(x_predict.shape)
print(x_predict)
# 벡터 한개는 1차원 
result = model.predict(x_predict)
print('loss : ',loss)
print('[[6],[7],[8],[9],[10]]의 결과 : ',result)
print('걸린시간 : ',round(end-start,2))

# [[6],[7],[8],[9],[10]]의 결과 :
