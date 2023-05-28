.py
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Input
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
#from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from tensorflow.keras.utils import to_categorical
# from keras.datasets import mnist #옛날 방식 텐서플로우 2.8부터 케라스가 분리되었다.
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# from tensorflow.keras.utils import to_categorical


# [실습] cnn의 성능 따라잡기
# 1. 데이터
# Dense로 바꾸기
# 여기서 reshape 해주기
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print("x_train",x_train.shape) # x_train(50000, 32, 32, 3)
print("x_test",x_test.shape)# x_test (10000, 32, 32, 3)
print("y_train : ",y_train.shape) # y_train :  (50000, 1)    
print("y_test : ",y_test.shape) #y_test :  (10000, 1)
print(np.unique(y_train)) #[0 1 2 3 4 5 6 7 8 9]


y_train= to_categorical(y_train)
y_test= to_categorical(y_test)


x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)
print("x_train",x_train.shape) # x_train(50000, 32, 32, 3)
print("x_test",x_test.shape)# x_test


#scaler=StandardScaler()
#scaler=MaxAbsScaler()
#scaler=MinMaxScaler()
#scaler=RobustScaler()
print("x_train : ",x_train.shape) # x_train :  (50000, 32, 32, 3)
print("x_test : ",x_test.shape)# x_test :(10000, 32, 32, 3)
print("y_train : ",y_train.shape) # y_train :   (50000, 10)
print("y_test : ",y_test.shape) #y_test :   (10000, 10)


# 2. 모델 구성
model = Sequential()
model.add(Dense(64,input_shape=(32*32*3,)))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10,activation='softmax'))
#64 = 유니츠
#model.add(Dense(64,input_shape=(28*28,)))


# 3. 컴파일 , 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30,mode='min',
                   verbose=1,restore_best_weights=True
                   )
model.fit(x_train,y_train,epochs=20,batch_size=30,validation_split=0.2,verbose=1,callbacks=[es])


# 4. 평가, 예측
result = model.evaluate(x_test,y_test)
print('result : ',result)
y_pred = model.predict(x_test)

acc= accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))
print('acc : ',acc)

#result :  0.4189961850643158
#acc :  0.8867
