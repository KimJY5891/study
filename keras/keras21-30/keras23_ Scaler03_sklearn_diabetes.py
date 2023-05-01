import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
#1.
datasets= load_diabetes()
x= datasets.data
y= datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print("x:",x.shape) #(442, 10)
print("y:",y.shape) #(442,)

#2.모델구성
model = Sequential()
model.add(Dense(10,input_dim=10))
model.add(Dense(12))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',verbose=1,
                   restore_best_weights=True)
model.fit(x_train,y_train,epochs=1220,batch_size=72,callbacks=[es])

#4.평가,예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
"""
NON
loss:  2341.106201171875
r2스코어 :  0.6029193680021552
"""
"""
scaler = StandardScaler()
loss:  2629.53466796875
r2스코어 :  0.5539983025703363
"""
"""
#scaler = MinMaxScaler()
loss:  2518.887451171875
r2스코어 :  0.5727654936133038
"""
"""
#scaler = MaxAbsScaler()
loss:  2470.78369140625
r2스코어 :  0.5809244558064177
"""
"""
scaler = RobustScaler()
loss:  2482.353759765625
r2스코어 :  0.578962019504945
"""
# none 승
