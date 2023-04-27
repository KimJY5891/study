
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#교육용 데이터셋
from sklearn.datasets import fetch_california_housing
#1. 데이터
datasets= fetch_california_housing()
x= datasets.data
y= datasets.target
print("x:",x.shape) #(20640,8)
print("y:",y.shape) #(20640,)
"""
[실습]
1.trainsize 0.7
2. r2 0.55 ~ 6이상
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715

)#1)train_size=0.7,random_state=257

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
"""
loss:  0.5869807600975037
r2스코어 :  0.556236925360099
"""
#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=740,batch_size=120,validation_split=0.2)
#1)loss='mse',optimizer='adam',epochs=4000,batch_size=32 777 77

#4.평가,예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
'''
loss:  15.7406644821167
r2스코어 :  0.824754581900729
'''
"""
    validation_split=0.2
    97/97 [==============================] - 0s 2ms/step - loss: 0.6330 - val_loss: 0.5626
194/194 [==============================] - 0s 961us/step - loss: 0.6071
loss:  0.607077419757843
194/194 [==============================] - 0s 920us/step
r2스코어 :  0.5410435671417215
"""
"""
    non-validation_split=0.2
    loss:  0.6044106483459473
r2스코어 :  0.5430596949779225
"""
