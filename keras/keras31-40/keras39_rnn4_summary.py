import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x= np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])

print(x.shape,y.shape)#(7, 3) (7,)
x=x.reshape(5,5,1)
print(x) #  [[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]], [[5],[6] ...
print(x.shape) #(5, 5, 1)

#2.모델 구성
model=Sequential()
model.add(SimpleRNN(1,input_shape=(5,1))) #input 모두가 행빼고 나머지 이듯
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
model.summary()

'''
DNN
(각 Layer의 파라미터 개수) = (input_dim X 노드수 + 노드수)
rnn
( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 + unit 개수 ) + ( 1(바이어스는 하나) * unit 개수)

unit *(feature +bias +units)=params
한 바퀴 돌아서 곱하기 때문에 이런 식이 나온다.
바이어으스는 왜 1인가 
200*200+1(인풋열)*200+200 = 40400
LSTM = rnn 계산의 4배
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 200)               40400     
                                                                 
 dropout (Dropout)           (None, 200)               0         
                                                                 
 dense (Dense)               (None, 8)                 1608      
                                                                 
 dense_1 (Dense)             (None, 16)                144       
                                                                 
 dense_2 (Dense)             (None, 32)                544       
                                                                 
 dense_3 (Dense)             (None, 64)                2112      
                                                                 
 dense_4 (Dense)             (None, 128)               8320      
                                                                 
 dense_5 (Dense)             (None, 256)               33024     
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_6 (Dense)             (None, 128)               32896     
                                                                 
 dense_7 (Dense)             (None, 64)                8256      
                                                                 
 dense_8 (Dense)             (None, 16)                1040      
                                                                 
 dense_9 (Dense)             (None, 4)                 68        
                                                                 
 dense_10 (Dense)            (None, 1)                 5         
                                                                 
=================================================================
Total params: 128,417
Trainable params: 128,417
Non-trainable params: 0
_________________________________________________________________
'''
