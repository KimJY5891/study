from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Flatten,Conv2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = fetch_california_housing()
x=datasets.data
y=datasets.target
print(x.shape,y.shape) #(20640, 8) (20640,)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8,random_state=333)
x_train = x_train.reshape(x_train.shape[0],2, 2, 2)
x_test = x_test.reshape(x_test.shape[0],2, 2, 2)
print(x_train.shape,x_test.shape) #(16512, 2, 2, 2) (4128, 2, 2, 2)

#2. 모델
# 시퀀셜 모델
model =Sequential()
model.add(Conv2D(
    filters=1,
    kernel_size=(2,2),
    input_shape=(13,1,1)))
model.add(Conv2D(
    filters=10,
    kernel_size=(2,2)))
model.add(Conv2D(
    filters=10,
    kernel_size=(2,2)))
model.add(Conv2D(
    filters=10,
    kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(13))

# model =Sequential()
# model.add(Dense(30,input_shape=(13,)))
# model.add(Dropout(0.3))
# model.add(Dense(20))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# 모델.이벨류 에이트에 들어가는 연산값에는 드랍아웃은 들어가지 않는다. 
# 가중치가 생성됐을 때

#함수형 모델 
# input01 = Input(shape=(13,)) #input01 = 레이어 이름 
# demse01 = Dense(30)(input01)
# demse02 = Dense(20)(demse01)
# demse03 = Dense(10)(demse02)
# output01 = Dense(1)(demse03)

#함수형 모델 
# model = Model(inputs=input01,outputs=output01)

#3. 컴파일 훈련 
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
                   verbose=1, #디폴트 0
                   restore_best_weights=True,
                   )

model.fit(x_train,y_train,
          epochs=1220,batch_size=72,
          callbacks=[es,],#mcp],
          validation_split=0.2
          )

#4.평가,예측
print('====================1. 기본출력 =================')
loss=model.evaluate(x_test,y_test,verbose=0)
print('loss: ',loss)
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
''''''
