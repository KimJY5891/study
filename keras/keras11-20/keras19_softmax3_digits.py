import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_digits()
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
print("y_test : ",y_test)
print('shape :',x.shape,y.shape)
print(np.unique(y_train,return_counts=True))

# 2. 모델 구성
model =Sequential()
model.add(Dense(50,activation="relu",input_dim=64))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(10,activation="softmax"))


# 3. 컴파일 훈련 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=10,mode='max',
               verbose=1,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=700,batch_size=800,verbose=1,validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)
