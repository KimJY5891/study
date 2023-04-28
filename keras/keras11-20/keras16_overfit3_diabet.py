#diabet:당뇨병
#어느 데이터셋인지도 알아내야함

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
#대회문제도 이정도 
"""
[실습]
1.trainsize 0.7~0.9
2.r2 0.62 이상
2번은 그만큼 교육용 데이터인데도 잘 안맞기도한다. 
실무용 데이터는 더 안맞을 수도 있다. 
잘맞게하려면 데이터 정제를 잘해야한다. 
"""
#1.
datasets= load_diabetes()
x= datasets.data
y= datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
print("x:",x.shape) #(442, 10)
print("y:",y.shape) #(442,)
#전체적 모양 보고 나서 데이터를 분석한다.
#2.모델구성
model = Sequential()
model.add(Dense(10,input_dim=10))
model.add(Dense(12))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
#노드나 레이어가 늘어날 수록 훈련 값을 더 높여야 잘 나오는 것으로 추측된다.

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
hist =model.fit(x_train,y_train,epochs=1000,batch_size=120,
                validation_split=0.2,
          verbose=1)
print(hist.history)
# 훈련 했던 값들은  model에 저장되어있다.
# model.fit은 결과치를 반환한다. 

#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)
#result =model.predict([17])



import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['val_loss'],marker='.',c='red',label='val_loss') # 뭔가 명시하지 않아도 된다는데 
plt.plot(hist.history['loss'],marker='.',c='blue',label='loss') # 뭔가 명시하지 않아도 된다는데 
plt.title('보스턴') #이름 지어주기
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()
