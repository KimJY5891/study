from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScalar #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리


datasets = load_boston()
x = datasets.data
y = datasets.target

print(type(x))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8,random_state=333    
)

scaler = StandardScalar()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#2. 모델
#시퀀셜 모델
# model =Sequential()
# model.add(Dense(30,input_shape=(13,)))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

#함수형 모델 
input01 = input(shape=(13,)) #input01 = 레이어 이름 
demse01 = Dense(30)(input01)
demse02 = Dense(20)(demse01)
demse03 = Dense(10)(demse02)
output01 = Dense(1)(demse03)

#함수형 모델 
model = Model(inputs=input01,outputs=output01)

#함수형 모델은 시작은 어디고 끝은 어딘지
# 차이는 위에서 모델을 정의하냐 마지막에 정의하는 것의 차이

# 데이터가 3차원이면(시계열 데이터)
# (1000, 100, 1) ->>> input_shape=(100, 1) 행 무시
# 데이터가 4차원이면(이미지 데이터)
# (60000, 32, 32, 3) ->>> input_shape=(32, 32, 3) 행 무시

#3. 컴파일, 훈련 
model.compile(loss='mse',optimize='adam')

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)
