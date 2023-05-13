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
path = "./_data/dacon_diabets/"
path_save = "./_save/dacon_diabets/"


# 1. 데이터

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################                                                     

train_csv = train_csv.dropna()

#####################train_csv 데이터에서 x와 y를 분리#######################
x = train_csv.drop(['Outcome'],axis=1)
y = train_csv['Outcome']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size = 0.2, shuffle = True, random_state=337, # stratify=y
)

# 2. 모델 구성

start = time.time()
model = Sequential()
model.add(Dense(5, input_shape = (x_train.shape[1],)))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='relu'))

# 함수 모델 
# input1 = Input(shape=(x_train.shape[1],))
# dense1 = Dense(5)(input1)
# dense2 = Dense(8,activation='relu')(dense1)
# dense3 = Dense(8,activation='relu')(dense2)
# dense4 = Dense(8,activation='relu')(dense3)
# dense5 = Dense(8,activation='relu')(dense4)
# output1 = Dense(1,activation='relu')(dense5)

# model = Model(inputs = input1, outputs = output1)


# 3. 컴파일 훈련 
model.compile(loss = 'mse', optimizer = 'adam')
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

y_predict = model.predict(x_test)

r2_score = r2_score(y_test, y_predict)
print("r2_score 스코어 : ", r2_score)
'''
Sequential 모델 
걸린 시간 :  13.365 초
loss :  0.3358778655529022
r2_score 스코어 :  -0.5057471264367812
'''
'''
함수 모델 
걸린 시간 :  12.617 초
loss :  0.2546629309654236
r2_score 스코어 :  -0.14165899866567644
'''

# 시간 : 함수 모델 
# 점수 : 함수 모델 
