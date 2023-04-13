import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from tensorflow.keras.datasets import fashion_mnist




(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape)) #((60000, 28, 28), (60000,)) ((10000, 28, 28), (10000,))


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
print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape)) # ((60000, 28, 28, 1), (60000, 10)) ((10000, 28, 28, 1), (10000, 10))




# 2. 모델 구성
model = Sequential()
#model.add(Dense(10,input_shape=(3,)))
model.add(Conv2D(
    filters=7,
    kernel_size=(2,2),
    input_shape=(28,28,1)))
model.add(Conv2D( # 위(n,7,7,7)에서 받아서 자르는것이다.
    filters=4, #이름 : 필터(예전:아웃풋)
    kernel_size=(3,3),#(n,6,6,4)
    activation='relu'# 웨이트가 마이너스 인것도 있다. 랜덤으로 해서 나왔을 때 마이너스 나온다면 렐루로 0으로 만들어준다. 모든 4차원의 값은
    )) #(n,5,5,4)
model.add(Conv2D(filters=10,
                kernel_size=(2,2)))#(n,5,5,10)
model.add(Flatten())#(n,4x4x10)
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()




#3.컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=30,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs= 16,batch_size=30,
                validation_split=0.2,verbose=1,callbacks=[es])


# 4. 평가, 예측
result =model.evaluate(x_test,y_test)
print('result : ',result )
y_predict=model.predict(x_test)
acc = accuracy_score(np.argmax(y_test),np.argmax(y_predict))
print('acc : ',acc)
