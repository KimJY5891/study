import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris,load_wine,load_digits
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(np.unique(y))
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2, shuffle = True, random_state=337,stratify=y
)
# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_shape = (x_train.shape[1],)))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(y_train.shape[1],activation='softmax'))

# 함수 모델 
# input1 = Input(shape=(x_train.shape[1],))
# dense1 = Dense(5)(input1)
# dense2 = Dense(8,activation='relu')(dense1)
# dense3 = Dense(8,activation='relu')(dense2)
# dense4 = Dense(8,activation='relu')(dense3)
# dense5 = Dense(8,activation='relu')(dense4)
# output1 = Dense(y_train.shape[1],activation='softmax')(dense5)

# model = Model(inputs = input1, outputs = output1)


# 3. 컴파일 훈련 
import time 
start= time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

model.fit(x_train, y_train,
                 epochs = 100,
                 batch_size = 32,
                 validation_split = 0.2,
                 verbose = 1,
                 # callbacks=[es],
                 )
               
end= time.time()


# 4. 평가 예측 

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_predict)
print("acc 스코어 : ", acc)
print('걸린시간 : ',round(end-start,2))
'''
Sequential 모델 
loss :  0.2947514057159424
12/12 [==============================] - 0s 2ms/step
acc 스코어 :  0.8888888888888888
걸린시간 :  16.27
'''
'''
함수 모델 
loss :  0.31823793053627014
12/12 [==============================] - 0s 2ms/step
acc 스코어 :  0.8805555555555555
걸린시간 :  15.44
'''

# 시간 : 함수모델 
# 스코어 : Sequential 모델 
