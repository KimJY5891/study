import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
docs = ['너무 재밋어요','참 최고예요','참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글세요',
        '별로에요','생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다','참 재밋네요','환희가 잘 생기긴 했어요',
        '환희가 안해요'
        ]

# 긍정 1, 부정 0 
label = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])
print(label.shape)
# 펼치기 전 까지 수치화 작업만 하기 

token = Tokenizer()
token.fit_on_texts(docs)
# docs가 이미 리스트 형{'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밋어요': 5,
# '최고예요': 6, '만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10,
# '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요
#': 16, '글세요': 17, '별로에요': 18, '생각보다': 19, '지루해요': 20, 
# '연기가': 21, '어색해요': 22, '재미없어요': 23, '재미없다': 24, '재밋네요': 25,
# '생기긴': 26, '했어요': 27, '안해요': 28}   태라서 리스트로 사용할 필요가 없다.
print(token.word_index)
# token.fit_on_texts([docs]) - 이미 리스트인데 리스트 또해서 잘 못한것
# {'너무 재밋어요': 1, '참 최고예요': 2, '참 잘 만든 영화예요': 3,
# '추천하고 싶은 영화입니다.': 4, '한 번 더 보고 싶네요': 5, '글세요': 6,
# '별로에요': 7, '생각보다 지루해요': 8, '연기가 어색해요': 9, '재미없어요': 10,
# '너무 재미없다': 11, '참 재밋네요': 12, '환희가 잘 생기긴 했어요': 13,
# '환희가 안해요': 14}
x = token.texts_to_sequences(docs)
print('x : ',x) 
# [[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]
# type:list
# 빵꾸난 자리 모두 0
# 어순 상으로 뒤가 중요한데 패딩을 0으로 해서 뒤를 0으로 채운다면 아무것도 아닌 0 중요하게 여겨짐
# 그래서 0은 앞으로 채워야 한다.

from tensorflow.keras.preprocessing.sequence import pad_sequences
# 순서가 있는 놈의  패드를 채운다

pad_x = pad_sequences(
    x,
    padding='pre', # 앞에서 부터 0을 채우겟다.
    maxlen=5, # 전체가 다 5개 크기가 된다. / 
    # 4일 경우 5짜리가 앞에 하나가 잘린다.  이유는 pre라서 앞으로 잘림
    # post는 뒤에가 잘림
)
print(pad_x)
print(pad_x.shape)  # (14, 5)
# [[ 0  0  0  2  5]
#  [ 0  0  0  1  6]
#  [ 0  1  3  7  8]
#  [ 0  0  9 10 11]
#  [12 13 14 15 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0  0 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0 21 22]
#  [ 0  0  0  0 23]
#  [ 0  0  0  2 24]
#  [ 0  0  0  1 25]
#  [ 0  4  3 26 27]
#  [ 0  0  0  4 28]]
word_size=len(token.word_index) # 단어 사전의 길이
print( 'word_size : ',word_size) #word_size :  28

pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)
print(pad_x.shape)


# 2. 모델 

model = Sequential()
model.add(Reshape(target_shape=(5,1), input_shape=(5,1))) # 데이터 리쉐이프 없이 할 수 있다
model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일,훈련

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=30,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(pad_x,label,epochs= 30,batch_size=16,
                validation_split=0.2,verbose=1,callbacks=[es])  

# 4. 평가, 예측

acc = model.evaluate(pad_x,label)[1]  # 로스와 매트릭스 값이 들어간다. 나는  acc만 뽑고 싶다. 
print('acc : ',acc )

# 문제점 ? 
# 데이터 자체를 변환할때 가치에 대한 의문
# 그래서 같은 값어치의 값은 원핫해줌
# 대부분 0이라면 데이터가 효율적이지 않다. 
# 그래서 나온게 임베딩
