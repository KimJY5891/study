from tensorflow.keras.datasets import imdb # 영화평
# 2.8버전부터는 텐서플로우랑 케라스를 분리한다고 할 수 도 있음
import numpy as np
import pandas as pd

(x_train,y_train),(x_test,y_test)=imdb.load_data(
    num_words=10000,
)
# 리스트 형태이고 안에 길이가 다 다르다.
print(x_train)
print(x_train.shape) #(25000,)
print(x_test.shape) #(25000,)
print(y_train) #[1 0 0 ... 0 1 0] -> 
print(y_train.shape) #(25000,)
print(np.unique(y_train,return_counts=True)) #[0 1] 
# 판다스에서 value_counts
# 만개와 백개가 있다면 백개짜리 증폭을 하도록 고려해야한다. 아니면 모델이 만개 쪽으로 많이 찾아내는 걸로 모델이 돌아감 
# return_counts=True -> 데이터의 갯수들을 찾아낼수 있다. 
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
#판다스 버전
print(pd.value_counts(y_train))
# 1    12500
# 0    12500
# dtype: int64

print('영화평의 최대 길이 : ', max(len(i) for i in x_train) ) #2494
print('영화평의 평균 길이 : ', sum(map(len,x_train))/len(x_train)) #238.71364

#[실습] 만들어보기
# 전처리 + 모델구성 +컴파일 훈련 + 
#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding = 'pre', maxlen = 100, 
                        truncating='pre') #padding 어딜채울거냐 truncating 어딜버릴거냐

print(x_train.shape) #(8982, 100)

x_test = pad_sequences(x_test, padding='pre', maxlen = 100,
                       truncating='pre')
print(x_test.shape) #(2246, 100)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM

pad_x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
pad_x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#2. 모델 구성 
model = Sequential()
model.add(Embedding(10000,32, input_length=100))
model.add(LSTM(32))
model.add(Dense(32))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일 훈련 
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])
model.fit(pad_x_train, y_train, epochs = 1)
#4. 평가 예측
acc = model.evaluate(pad_x_test, y_test)[1]
print(acc)


