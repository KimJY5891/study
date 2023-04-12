import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,Dropout,LSTM

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x= np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
y= np.array([6,7,8,9,10])
print(x.shape,y.shape)
x=x.reshape(5,5,1)
print(x)#[[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]], [[5],[6] ...
print(x.shape) #(5, 5, 1)

#2.모델 구성
model=Sequential()
model.add(LSTM(200,input_length=5,input_dim=1)) #가독성이 떨어진다.
# [batch,input_length,input_dim]
# [batch,timesteps,feature]
#model.add(LSTM(200,input_shape=(5,1))) #input 모두가 행빼고 나머지 이듯 
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

"""
params = dim(W)+dim(V)+dim(U) = n*n + kn + nm
# n - dimension of hidden layer
# k - dimension of output layer
# m - dimension of input layer

_______________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 200)               161600    
                                                                 
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
Total params: 249,617
Trainable params: 249,617
Non-trainable params: 0
_________________________________________________________________
"""
