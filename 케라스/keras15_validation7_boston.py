
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
#교육용 데이터셋
from sklearn.datasets import load_boston
#1. 데이터
datasets= load_boston()
#정규화 :예를 들어서 1에서 1조 데이터일때, 1조x1조 경우, 오버 쿨럭?걸릴수도 있음 
#그래서 0부터 1사이로 압축 => 최대치로 나눈다.

x= datasets.data
y= datasets.target
print('feature_names')
print("x:",x)
print("y:",y)
#워닝 = 경고는 하지만 돌아는간다.
#에러 = 실행안됌실행안됌

#'feature_names': array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
#       'TAX', 'PTRATIO', 'B', 'LSTAT']  'B' - 그 당시는 문제 없었지만 지금은 인종차별적인 문제가 있다. 
#열 13개 input_dim= 13

print(datasets.DESCR)
#INSTANCE(506) -예시라는 말은 데이터라는 것도 의미할수도 있다. 
#ATTRIBUTE(13) - 속성 - 특성, 열
#MEDV = 결과값 =Y 단위 천달라
print("x:",x.shape)#(506,13)
print("y:",y.shape)#(506,)(벡터는 1개)

"""
[실습]
1.trainsize 0.7
2. r2 0.8이상
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=2579
)#1)train_size=0.7,random_state=2579
#2.모델구성
model = Sequential()
model.add(Dense(26,input_dim=13))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(26))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(1))
#1)13 26 50 100 1000 100 50 26 7 5 1

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=3000,batch_size=32,validation_split=0.2)
#1)loss='mse',optimizer='adam',epochs=4000,batch_size=32

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
    9/9 [==============================] - 0s 6ms/step - loss: 27.3358 - val_loss: 25.4455
loss:  20.29852867126465
5/5 [==============================] - 0s 1ms/step
r2스코어 :  0.774010556270542
"""
"""
   non-validation_split=0.2 
   loss:  24.302047729492188
    r2스코어 :  0.7294381848569711   
"""
