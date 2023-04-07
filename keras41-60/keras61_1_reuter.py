from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

# 1. 데이터
(x_train,y_train),(x_test,y_test) =reuters.load_data(
    num_words=10000, # 단어사전 갯수 지정가능 상위 만 개 
    test_split=0.2 # 
)
print(x_train) #이미 수치화된 데이터
print(y_train) # [ 3  4  3 ... 25  3 25]
print(x_train.shape,y_train.shape) # (8982,) (8982,)
print(x_test.shape,y_test.shape) # (2246,) (2246,)

# 파이썬 버전 : 3.9.12 , tensorflow-gpu : 2.7.4, 아나콘다.

print(len(x_train[0]),len(x_train[1])) # 87 56 => 넘파이 안에 리스트 형식으로 들어잇음
'''
print(np.unique(y_train)) # 46개 0 ~ 45
print(type(x_train),type(y_train)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]),type(y_train[0])) # <class 'list'> <class 'numpy.int64'>
'''
print('뉴스 기사의 최대 길이 : ', max(len(i) for i in x_train) ) #2376
print('뉴스 기사의 평균 길이 : ', sum(map(len,x_train))/len(x_train)) #145.5398574927633

# 전처리 패드 시퀀스
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(
    x_train,padding='pre', #앞을 채움
    maxlen=100,
    truncating='pre' #버리는 놈은 잎에서 잘라서 버릴거야 
)
print(x_train.shape)# (8982, 100)

# 나머지는 전처리하고
# 2. 모델 구성

# 3. 
# 내일부터 프로젝트 
# 목요일부터 시작발표 컨펌  빨리 바든넥 조흠
# 발표 5분 
# pt 5장
# 에이아이팩토리 3개 
# 신문의 기사의 종류를 맞추는것 

#전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(x_train, padding = 'pre', maxlen = 100, 
                        truncating='pre') #padding 어딜채울거냐 truncating 어딜버릴거냐

print(x_train.shape) #(8982, 100)

x_test = pad_sequences(x_test, padding='pre', maxlen = 100,
                       truncating='pre')

print(x_test.shape) #(2246, 100)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM

pad_x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
pad_x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
#2. 모델 구성 
model = Sequential()
model.add(Embedding(10000,32, input_length=100))
model.add(LSTM(32))
model.add(Dense(32))
model.add(Dense(46, activation = 'softmax'))
#3. 컴파일 훈련 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['acc'])
model.fit(pad_x_train, y_train, epochs = 1)
#4. 평가 예측
acc = model.evaluate(pad_x_test, y_test)[1]
print(acc)

