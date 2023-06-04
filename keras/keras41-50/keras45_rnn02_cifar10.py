
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten ,LSTM
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리


# 1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data() # 데이터를 넣어줌
print("x_train.shape : ",x_train.shape) #(60000, 28, 28)
print("y_train.shape : ",y_train.shape)
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




# 2. 모델 구성


model = Sequential()
#model.add(Dense(10,input_shape=(3,)))
model.add(LSTM(32, input_shape=(28,28)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))




#3.컴파일,훈련


model.compile(loss='categorical_crossentropy',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=30,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs= 16,batch_size=25000,
                validation_split=0.2,verbose=1,callbacks=[es])


# 4. 평가, 예측


result =model.evaluate(x_test,y_test)
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)


import matplotlib.pyplot as plt
plt.imshow(x_train[3333],'gray') # 그림 보여줌
plt.show()
