import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리


#1. 데이터
datasets = load_wine()
print(datasets.DESCR) #판다스 describe()
print(datasets.feature_names) # 판다스 clolums
x=datasets['data']
y=datasets.target
print('shape :',x.shape,y.shape) #(178, 13) (178,)
print(x)
print(y) 
print("y의 라벨 값 :",np.unique(y))
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    train_size=0.8,
    shuffle=True,random_state=456,
    stratify= y 
)

#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("y_test : ",y_test)
print(np.unique(y_train,return_counts=True))

# 2. 모델 구성
model =Sequential()
model.add(Dense(50,activation="relu",input_dim=13))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(3,activation="softmax"))


# 3. 컴파일 훈련 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=10,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=700,batch_size=800,verbose=1,validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

"""
NON
result :  [1.1085851192474365, 0.6388888955116272]
acc :  0.61111111111111
"""
"""
scaler = StandardScaler()
result :  [0.4472161829471588, 0.8888888955116272]
acc :  0.777777777777777
"""
"""
#scaler = MinMaxScaler()
result :  [0.7328977584838867, 1.0]
acc :  0.4166666666666667
"""
"""
#scaler = MaxAbsScaler()
result :  [0.8579521179199219, 0.8888888955116272]
acc :  0.05555555555555555
"""
"""
scaler = RobustScaler()
result :  [0.8564956784248352, 0.7777777910232544]
acc :  0.2222222222222222
"""
