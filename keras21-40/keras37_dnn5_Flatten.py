import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

# 1. epdlxj 
(x_train,y_train),(x_test,y_test) = mnist.load_data() # 데이터를 넣어줌 

scaler = MinMaxScaler()
x_train = x_train.reshape(-1,1)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)
print("x_train.shape01: ",x_train.shape)
print("y_train.shape01 : ",y_train.shape)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1) # 데이터의 구조만 바귀는 것 순서와 값이 바뀌는 게 아님 
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)


# 2. 모델 구성
model = Sequential() 
# [batch, timesteps, feature]
# [배치단위,자른 단위 수 ,자른 데이터,열특성 ]
model.add(Dense(10,input_shape=(28,28)))
model.add(Dense(9))
model.add(Dense(8))
model.add(Flatten())
model.add(Dense(7))
model.add(Dense(10, activation='softmax'))
model.summary() 
#3차원받아들이고 2차원으로 바꿔줘야한다. 
#flatten으로 변화 해주면 된다.
# 
'''
#3.컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=30,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs= 16,batch_size=25000,
                validation_split=0.2,verbose=1,callbacks=[es])

#4. 평가 예측
# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

import matplotlib.pyplot as plt
plt.imshow(x_train[3333],'gray') # 그림 보여줌
plt.show()
'''
