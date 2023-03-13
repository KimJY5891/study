#강제로 나쁘게 만들자
# 그래야 좋게 만드는 것을 이해할 수 있다. 
"""
조건
1) r2를 음수가 아닌 0.5이하로 만들것
2) 데이터는 건들지 말것
3) 레이어는 인풋푸아웃풋 포함해서 7개 이상
4) batch_size=1  
5) 히든 레이어의 노드는 10개 이상 100개 이하(한레이어당)
6) train_size=0.75
7) epoch 100번이상
8) loss지표는 mse, mae
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.medel_selection import train_test_split

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,6,17,23,21,20])
x_train, x_test, y_train, y_test = train_test_split(
x,#x_train과 x_test로 분리
y,#y_train과 y_test로 분리
train_size=0.75, shuffle=True, random_state
=2569)

# 2. 모델구성
model=Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))
#1000-> 40 언저리
#r2스코어 :0.41977856142668624 | loss : 8.726530075073242
#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer="adam")
model.fit(x_train,y_train,epochs=100, batch_size=1)

#4. 평가 예측
loss=model.evaluate(x_test,y_test)
print('loss : ',loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2 스코어 : ',r2)

"""
r2는 보조 지표이다.
그래서 로스와 서로 값이 다를 때 로스를 먼저 믿는다.
r2가 마이너스 나오면 별로이다. 
r2는 상대적인 지표이다.
"""
