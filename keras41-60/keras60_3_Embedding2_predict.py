import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Reshape,Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
docs = ['너무 재밋어요','참 최고예요','참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글세요',
        '별로에요','생각보다 지루해요','연기가 어색해요',
        '재미없어요','너무 재미없다','참 재밋네요','환희가 잘 생기긴 했어요',
        '환희가 안해요']

################################ [실습] ################################
x_predict = '나는 성호가 정말 재미없다'
# 긍정인지 부정인지 맞춰봐
# 얘만 다로 texts_to_sequences 넣고 나서 하면 다르다
# DOCS와 같은 수치를 잡아야하는데,
#????????????????????????

# 긍정 1, 부정 0 
label = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])
print(label.shape)

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밋어요': 5,
# '최고예요': 6, '만든': 7, '영화예요': 8, '추천하고': 9, 
# '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, 
# '보고': 15, '싶네요': 16, '글세요': 17, '별로에요': 18, 
# '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22,
# '재미없어요': 23, '재미없다': 24, '재밋네요': 25, 
# '생기긴': 26, '했어요': 27, '안해요': 28}
x = token.texts_to_sequences(docs)
# x = token.texts_to_sequences(docs,[x_predict]) ->둘이 들어가면 됌 문법은 틀릴수도??
print('x : ',x) 
#[[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], 
# [12, 13, 14, 15, 16], [17], [18], [19, 20], 
# [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]

token_x_predict = Tokenizer()
# token_x_predict.fit_on_texts([x_predict])-> 틀림 
print('token_x_predict.word_index: ',token_x_predict.word_index)
#{'나는': 1, '성호가': 2, '정말': 3, '재미없다': 4}
x_predict = token_x_predict.texts_to_sequences([x_predict]).copy()
print(x_predict) #[[1, 2, 3, 4]]
from tensorflow.keras.preprocessing.sequence import pad_sequences
maxlen= 5
padding = 'pre'
pad_x = pad_sequences(
    x,
    padding=padding,
    maxlen=maxlen,
)
pad_x_predict = pad_sequences(
    x_predict,
    padding=padding,
    maxlen=maxlen,
)
print(pad_x)
print(pad_x.shape)   #  (14, 5)
print(pad_x_predict) # [[0 1 2 3 4]]
print(pad_x_predict.shape)  # (1, 5)
word_size=len(token.word_index) # 단어 사전의 길이
print( 'word_size : ',word_size) # word_size :  28

pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)
pad_x_predict = pad_x_predict.reshape(pad_x_predict.shape[0],pad_x_predict.shape[1],1)
print(pad_x.shape)


# 2. 모델 

model = Sequential()
model.add(Embedding(28,10))
model.add(LSTM(28))
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

y_pred = np.round(model.predict(x_predict))
print(y_pred) #[[1.]] 
