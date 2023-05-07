from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split 
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = fetch_california_housing()
x=datasets.data
y=datasets.target
print('d')
print(x.shape,y.shape) #(20640, 8) (20640, )
x_train, x_test, y_train, y_test= train_test_split(
    x,y,train_size=0.8,random_state=333
)
#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print('dd')
print(np.min(x_train),np.max(x_train)) #-2.3805947933191054 113.55761057542182
print(np.min(y_train),np.max(y_train))

#2. 모델 구성
model = Sequential()
model.add(Dense(4,input_dim=8))
model.add(Dense(2))
model.add(Dense(8))
model.add(Dense(80))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(40))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.1))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
hist =model.fit(x_train,y_train,epochs=1000,batch_size=120,
          validation_split=0.2,verbose=1)
print(hist.history) 
#4. 평가, 예측 
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

"""
scaler = StandardScaler()
loss :  0.6204870343208313
"""
"""
#scaler = MinMaxScaler()
loss :  0.5746297240257263
"""
"""
#scaler = MaxAbsScaler()

"""
"""
scaler = RobustScaler()

"""
