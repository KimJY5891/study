import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
datasets = load_digits()
print(datasets.DESCR) #판다스 describe()
print(datasets.feature_names) # 판다스 clolums
x=datasets['data']
y=datasets.target
print('shape :',x.shape,y.shape) #(178, 13) (178,)
print("y의 라벨 값 :",np.unique(y)) #y의 라벨 값 : [0 1 2 3 4 5 6 7 8 9]
y = to_categorical(y)
print(x.shape,y.shape) #(1797, 64) (1797, 10)
x=x.reshape(1797,64,1)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,random_state=456,
    stratify= y
)
print("y_test : ",y_test)
print('shape :',x.shape,y.shape) #shape : (1797, 64) (1797, 10)
print(np.unique(y_train,return_counts=True))




# 2. 모델 구성
Input01=Input(shape=(64,1))
Conv1D01=Conv1D(10,2,padding='same')(Input01)
Conv1D02=Conv1D(10,2,padding='same')(Conv1D01)
Flatten01=Flatten()(Conv1D02)
Dense01 = Dense(12)(Flatten01)
Dense02 = Dense(12)(Dense01)
Output01= Dense(10,activation='softmax')(Dense02)
model=Model(inputs=Input01,outputs=Output01)




# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_acc',patience=10,mode='max',
               verbose=1,restore_best_weights=True,)
mcp=ModelCheckpoint(monitor='loss',mode='min',verbose=1,save_best_only=True,
               filepath='./_save/MCP/keras48_Conv1D_03_diabets_mcp.hdf5')
hist = model.fit(x_train,y_train,epochs=700,batch_size=800,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es,mcp])


# 4. 평가, 예측
result =model.evaluate(x_test,y_test)
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)
'''
result :  [0.2441631704568863, 0.9277777671813965]
acc :  0.9166666666666666
'''
