#api 가지고 올때 최상단에 몰빵해서 작성하면 가독성이 좋아진다.
#import 가져올때 순서는 상관없다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#실스반들기 [6]을 예측한다.

#2. 모델

model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=40)

#4. 평가 예측
loss = model.evaluate(x,y)
#x와 y의 값으로 생성된 가중치(w)를 넣어서 다시 판단하는 것
print("loss : ",loss)
result = model.predict([6])
print("[6]의 예측값 : ",result)
