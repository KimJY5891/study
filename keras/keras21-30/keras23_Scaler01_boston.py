#넘파이는 부동소수점 연산(실수)에 좋다.
#정규화 : 0부터 100만까지의 숫자를 0부터 1까지의 숫자로 바꿔줄 수 있다.  (100만으로 나누기)
#우리가 받는 데이터는 항상 x와 y값을 받고 시작한다. (그래야만 한다.)
#정규화 = nomalization (원래 단어 뜻은 다름)
#모든 x데이터를 0에서 1 사이로 만들어버린다 (y는 제외!!)
#과부화걸리지 않게함, 속도를 빠르게함, 성능도 좋아질 '수도' 있다**
#x가 나누어지더라도 가르치는 y는 동일하다. 
#(x-min)/(max-min)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']
# print(type(x))


# print(np.min(x), np.max(x))                    #x의 최소값 보기
# scaler= MinMaxScaler()
# scaler.fit(x)                 #변환할 준비를 해라
# x = scaler.transform(x)       #변환해라
# print(np.min(x), np.max(x))
print(type(x))     # <class 'numpy.ndarray'>
print(x)


# (사이킷런 1.2부터 임포트 안됨)
# pip uninstall scikit-learn
# pip install scikit-learn==1.1

print(np.min(x), np.max(x))     # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_test), np.max(x_test))     # 0.0 1.0

# 일반적으로 훈련 데이터만 정규화한다. 그 이유는 1이상의 값을 predict 할수도 있기때문
# 전체 데이터를 정규화 하지 않는 이유는 실제로 해보면 과적합되는 경우가 많기 때문이다.
# 훈련데이터(x_train)를 정규화 한 후, 훈련데이터의 정규화 '비율'에 맞춰서 test data를 변환시킨다.
# # 또한, x_predict 또한 비율에 맞춰서 변환시킨다. x_predict값은 1을 초과할수도, 0 미만일수도 있다.
# # 때문에 train_test_split 이후에 train에만 정규화를 시행한다.

# 2. 모델
model = Sequential()
model.add(Dense(1, input_dim=13))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
