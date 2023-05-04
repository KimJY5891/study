from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import to_categorical
 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
# 사이킷런의 스케일러는 2차원이여야한다.
# 하려면 4차원 -> 2차원 -> 4차원
# 스케일러 사용할 때 2차원만 사용가능
(x_train,y_train),(x_test,y_test) = cifar10.load_data() # 데이터를 넣어줌
print("x_train.shape : ",x_train.shape)
print("x_test.shape: ",x_test.shape)
print("y_train.shape : ",y_train.shape)
print("y_test.shape : ",y_test.shape)
# x_train.shape :  (50000, 32, 32, 3)
# x_test.shape:  (10000, 32, 32, 3)
# y_train.shape :  (50000, 1)
# y_test.shape :  (10000, 1)
# 4차원 데이터라서 리쉐이프 할 피요 없음


#scaler = MinMaxScaler()
x_train = x_train/255.
x_test = x_test/255.
# 이렇게 사용하면 바로 스케일링 가능  minNMaxScaler,
print(np.max(x_train),np.min(x_train)) # 1.0 0.0
# 이미지는 이렇게 할 수 있다.ㅅ
#최대 값 - 최소값
print(np.unique(y_train,return_counts=True)) #([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 이번 자료는 리쉐이프 할 필요 없다.
y_train= to_categorical(y_train)
y_test= to_categorical(y_test)


#2. 모델
model = Sequential()
model.add(Conv2D(
    10,(2,2),padding='same',input_shape=(32,32,3)
))
model.add(MaxPooling2D())
model.add(Conv2D(32,2))
model.add(MaxPooling2D())
model.add(Conv2D(32,2))
model.add(MaxPooling2D())
model.add(Conv2D(32,2))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,2))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,2))
# model.add(MaxPooling2D())
# model.add(Conv2D(32,2))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(10,activation='softmax')) acc0.62


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=30, mode='max',
                    verbose=1, restore_best_weights=True
                   )
model.fit(x_train,y_train,epochs=200,batch_size=1000,
          validation_split=0.2,verbose=1, callbacks=[es])


#4. 평가, 예측
result = model.evaluate(x_test,y_test)
print('result : ',result)  


y_predict=model.predict(x_test)
acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1))


# acc=accuracy_score(np.argmax(y_test,axis=1),np.argmax(y_predict,axis=1))
print("acc : ",acc )
# 0.75
