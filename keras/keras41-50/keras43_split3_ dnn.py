import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,Dropout,LSTM


dataset = np.array(range(1,101))
timesteps = 5
x_predict = np.array(range(96,106))
def split_x(dataset, timesteps) :
    list =[]
    for i in range(len(dataset)-timesteps +1) :
        subset = dataset[i:(i+timesteps)]
        list.append(subset)
    return np.array(list)


dataset= split_x(dataset,timesteps)
print(dataset)
print(dataset.shape) #(6,4)
x=dataset[:,:-1]
y=dataset[:,-1]


print(x_predict)
print(x_predict.shape)


#x_predict=x_predict[:,:-1]
 #(7, 4)


# 4개씩 잘랐다.


#X=bbb[:,:4]
print(x)
print(y)
print(x.shape) #(97, 3)
print(y.shape) #(97,)
#x= x.reshape(96,4,1)


# 2. 모델 구성
model= Sequential()
#model.add(LSTM(200,input_length=3,input_dim=1))
model.add(Dense(20,activation='relu',input_shape=(4,)))
model.add(Dense(200))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dense(16,activation='relu'))
model.add(Dense(32))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
import time
start=time.time()
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=45,mode='min',verbose=1,restore_best_weights=True)
model.fit(x,y,epochs=1200,batch_size=32,callbacks=[es])
end=time.time()


#4. 평가 예측

loss=model.evaluate(x,y)
print('loss : ',loss)

x_predict = split_x(x_predict,4)
print(x_predict.shape) #(6, 5)
print(x_predict)

result=model.predict(x_predict)
print('x_predict 결과: ',result)
print('걸린 시간 : ',round(end-start,2))
'''
loss :  323.3116149902344
(7, 4)
[[ 96  97  98  99]
 [ 97  98  99 100]
 [ 98  99 100 101]
 [ 99 100 101 102]
 [100 101 102 103]
 [101 102 103 104]
 [102 103 104 105]]
1/1 [==============================] - 0s 135ms/step
x_predict 결과:  [[69.3706  ]
 [70.05735 ]
 [70.7441  ]
 [71.43085 ]
 [72.117584]
 [72.80434 ]
 [73.49107 ]]
걸린 시간 :  8.53

'''
