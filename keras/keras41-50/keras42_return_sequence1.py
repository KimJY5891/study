# 컴플루션을 할 때 마다 성능이 좋아진다.???????
# cnn 3번이상해줘야함
# lstm은 3차원에서 사용해서 2차원으로 나옴
# 그래서 lstm을 연속으로 못 쓴다.
# 레이어도 리쉐이프 레이어가 있다.
# 다시 3차원으로 만들어서 주면 lstm을 또 사용할 수 있다.
# return_sequences - 순서를 다시 돌려준다.
# return_sequences = True  - 3차원으로 나옴
# return_sequences = False - 디폴트, 2차원 나옴
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN,Dropout,LSTM, GRU


#1. 데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
           ,[5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]) #아워너 80


#만들자 시작
print(x.shape,y.shape) #(13, 3) (13,)
# 리쉐이프
x=x.reshape(13,3,1)
print(x.shape,y.shape) #(13, 3,1) (13,)


#2. 모델 구성
model= Sequential()
#model.add(LSTM(200,input_length=3,input_dim=1))
#model.add(LSTM(20,activation='relu',input_shape=(3,1)))
model.add(LSTM(200,activation='relu',input_shape=(3,1),return_sequences=True)) #디폴트는 false
#ndim = 숫자, 숫자= 차원,LSTM 두 개 이상 사용할 때 이용 return_sequences=True
model.add(LSTM(100,return_sequences=True))
model.add(GRU(36))
model.add(Dense(1))
model.summary()
"""
lstm (LSTM)                 (None, 3, 10)             480
lstm_1 (LSTM)               (None, 10)                840
dense (Dense)               (None, 1)                 11
"""
#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')
import time
start=time.time()
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=45,mode='min',verbose=1,restore_best_weights=True)
model.fit(x,y,epochs=2000,batch_size=32,callbacks=[es])
end=time.time()


#4. 평가 예측
loss=model.evaluate(x,y)
x_predict = np.array([50,60,70]).reshape(1,3,1)
print(x_predict.shape)
print(x_predict)

result=model.predict(x_predict)
print('loss : ',loss)
print('[50,60,70]의 결과: ',result)
print('걸린 시간 : ',round(end-start,2))
'''
loss :  20.252140045166016
[50,60,70]의 결과:  [[54.67582]]
걸린 시간 :  76.93
'''
