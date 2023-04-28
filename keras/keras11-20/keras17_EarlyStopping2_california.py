
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#교육용 데이터셋
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
#1. 데이터
datasets= fetch_california_housing()
x= datasets.data
y= datasets.target
print("x:",x.shape) #(20640,8)
print("y:",y.shape) #(20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715
)

#2.모델구성
model = Sequential()
model.add(Dense(4,input_dim=8))
model.add(Dense(2))
model.add(Dense(8))
model.add(Dense(80))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=50,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs=1000,batch_size=100,
          validation_split=0.2,callbacks=[es],verbose=1,
           )
print("히스토리 : ",hist.history)


import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
figsize=(9,6)
plt.plot(hist.history['val_loss'],marker='.',c='red',label='val_loss')
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.title('캘리포니아')
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()

# epoch 00 early stopping
# 훈련 했던 값들은  model에 저장되어있다.
# model.fit은 결과치를 반환한다. 

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
y_predict = model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 스코어 :',r2)
