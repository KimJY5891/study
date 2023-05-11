# 6만, 784,1 
# 6만,28,28
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten ,LSTM,MaxPooling2D,Conv1D, Reshape
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

# 1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data() # 데이터를 넣어줌 
x_train=x_train.reshape(60000,28,28,1)/255.
x_test=x_test.reshape(10000,28,28,1)/255.
#스케일링

print("x_train.shape : ",x_train.shape)#(60000, 28, 28, 1)
print("y_train.shape : ",y_train.shape) #y_train.shape :  (60000,)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1) 
# y_train= to_categorical(y_train)
# y_test= to_categorical(y_test)


# 2. 모델 구성

model = Sequential()
#model.add(Dense(10,input_shape=(3,)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',input_shape=(28,28,1)))
model.add(Conv2D(32,(3,3)))
model.add(Conv2D(10,3))
model.add(MaxPooling2D())
model.add(Flatten()) # (N,250)
model.add(Reshape(target_shape=(25,10))) #(n,25,10) 행인 데이터 갯수는 작성하지 않는다. 
# 리쉐이프 시,안에 있는 내용 순서는 바뀌지 않는다.
model.add(Conv1D(10,3,padding='same'))
model.add(LSTM(784))
model.add(Reshape(target_shape=(28,28,1))) #(n,25,10)
model.add(Conv2D(32,3,padding='same'))
model.add(Reshape(target_shape=(28*28))) 
model.add(Flatten()) #
model.add(Dense(10,activation='softmax')) 
# model.add(Dense(100))
# model.add(Dense(10,activation='softmax'))
# 리쉐이프 부분만 살려두기
model.summary()


'''
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
'''
