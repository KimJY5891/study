# 모델 함수형
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
#from keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from tensorflow.keras.utils import to_categorical
# 1. 데이터


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
print("x_train : ",x_train.shape) #x_train :  (60000, 28, 28, 1)
print("x_test : ",x_test.shape) # x_test :  (10000, 28, 28, 1)
print("y_train : ",y_train.shape) # y_train : (60000,)
print("y_test : ",y_test.shape) # y_test :  (10000,)
print(y_test)
print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)
print(y_test)
print("x_train : ",x_train.shape) # x_train : (60000, 28, 28, 1)
print("x_test : ",x_test.shape)# x_test :  (10000, 28, 28, 1)
print("y_train : ",y_train.shape) # y_train :  (60000, 10)
print("y_test : ",y_test.shape) # y_test :  (10000, 10)


#CNN인데 to_categorical 사용해야하는 이유?


#2. 모델


# model =Sequential()
# model.add(Conv2D(10,(2,2),padding='same',input_shape=(28,28,1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filter=64,kernel_size=(2,2),
#                  padding='valid',activation='relu'))
# model.add(Conv2D(32,2))
# model.add(Flatten())
# model.add(Dense(10,activation='softmax'))
# model.summary()
# inputs = Input(shape=(28, 28, 1))


input_01=Input(shape=(28,28,1))
Conv2D_01 =Conv2D(10,(2,2),padding='valid',activation='relu')(input_01)
MaxPooling2D_01=MaxPooling2D()(Conv2D_01)
Conv2D_02=Conv2D(64,(2,2),padding='valid',activation='relu')(MaxPooling2D_01)
Conv2D_03=Conv2D(32,(2,2))(Conv2D_02)
Flatten_01=Flatten()(Conv2D_03)
Outputs=Dense(10,activation='softmax')(Flatten_01)
model= Model(inputs=input_01,outputs=Outputs)


#3.컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=30,mode='min',
               verbose=1,restore_best_weights=True)
# 모델 체크 포인트
# 얼리스타핑과 모델 체크 포인트 - 히스토리에서 가져와서 멈추게 하는것이다.


model.fit(x_train,y_train,epochs= 16,batch_size=30,
                validation_split=0.2,verbose=1,callbacks=[es])


# 4. 평가, 예측
result =model.evaluate(x_test,y_test)
print('result : ',result )
y_pred=model.predict(x_test)
acc= accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))
print('acc : ',acc)
