from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=20580)

# [실습]
#  R2 0.62 이상

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=10, 
          # validation_split=0.2
          )

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

'''
validation_split=0.2 일 경우 
/step - loss: 1980.1715
loss :  1980.1715087890625
r2 스코어 :  0.6815945119298177
'''
'''
validation_split 없을 경우 
/step - loss: 1912.1865
loss :  1912.1865234375
r2 스코어 :  0.6925263041350322
'''
