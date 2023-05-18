import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import MinMaxScaler
import time

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(y)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2, shuffle = True, random_state=337,stratify=y
)

# 2. 모델 구성
start = time.time()
model = Sequential()
model.add(Dense(5, input_shape = (x_train.shape[1],)))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(y_train.shape[1],activation='softmax'))

# # 함수 모델 
# input1 = Input(shape=(x_train.shape[1],))
# dense1 = Dense(5)(input1)
# dense2 = Dense(8,activation='relu')(dense1)
# dense3 = Dense(8,activation='relu')(dense2)
# dense4 = Dense(8,activation='relu')(dense3)
# dense5 = Dense(8,activation='relu')(dense4)
# output1 = Dense(y_train.shape[1],activation='softmax')(dense5)

# model = Model(inputs = input1, outputs = output1)



# 3. 컴파일 훈련 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss',
              patience =20,
              mode = 'min',
              verbose=1,
              restore_best_weights=True,
              )
model.fit(x_train, y_train,
                 epochs = 100,
                 batch_size = 32,
                 validation_split = 0.2,
                 verbose = 1,
                 # callbacks=[es],
                 )
               
end = time.time()
print(' 걸린 시간 : ',round(end-start,3),'초')

# 4. 평가 예측 

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test, y_predict)
print("acc 스코어 : ", acc)
'''
Sequential 모델 
3632/3632 [==============================] - 6s 2ms/step - loss: 0.6455
loss :  0.6454960107803345
acc 스코어 :  0.7168489625913272
걸린 시간 :  7587.053
'''
'''
함수 모델 
걸린 시간 :  7587.053 초
3632/3632 [==============================] - 13s 4ms/step - loss: 0.6163
loss :  0.6162505745887756
acc 스코어 :  0.7289140555751573
'''
