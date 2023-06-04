from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten ,LSTM, Bidirectional, SimpleRNN, GRU
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler




# 1. 데이터


x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
           ,[5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]) #아워너 80
print(x.shape) #(13, 3)
print(y.shape) #(13,)
print(x_predict.shape) #(3,)
x=x.reshape(13,3,1)


# 2. 모델 구성


model=Sequential()
model.add(Bidirectional(LSTM(10,return_sequences=True),input_shape=(3,1)))
model.add(Bidirectional(LSTM(10,return_sequences=True)))
model.add(LSTM(10))
model.add(Dense(10))
model.add(Dense(1))




# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
import time
start=time.time()
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=200,mode='min',verbose=1,restore_best_weights=True)
model.fit(x,y,epochs=2000,batch_size=32,callbacks=[es])
end=time.time()


# 4. 평가, 예측
loss=model.evaluate(x,y)
x_predict = np.array([50,60,70]).reshape(1,3,1)
result=model.predict(x_predict)
print('loss : ',loss)
print('[50,60,70]의 결과: ',result)
print('걸린 시간 : ',round(end-start,2))
