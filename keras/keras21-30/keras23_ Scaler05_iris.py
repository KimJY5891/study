import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score #평가용 #r2는 회귀, accuracy는 분류모델에서 사용하고 가장 디폴트적인 것
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
x=datasets['data']
y=datasets.target
print(x.shape,y.shape) #(150, 4) (150,)
print(x)
print(y)
print("y의 라벨 값 :",np.unique(y)) #[0 1 2]
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
model.add(Dense(50,activation="relu",input_dim=4))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(3,activation="softmax"))

# 3. 컴파일 훈련 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=40,mode='max',
                   verbose=1,restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=700,batch_size=800,verbose=1,validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
print('loss : ',result[0] )
print('acc : ',result[1] )
y_predict=model.predict(x_test) #위치에 대한 반환
y_test_acc = np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=-1)
print('y_test_acc : ',y_test_acc)
print('y_predict : ',y_predict )
acc = accuracy_score(y_test_acc,y_predict)
print('accuracy_score : ',acc)

"""
NON
accuracy_score :  0.9666666666666667
"""
"""
scaler = StandardScaler()
result :  [0.12727436423301697, 0.9666666388511658]
loss :  0.12727436423301697
acc :  0.9666666388511658
"""
"""
#scaler = MinMaxScaler()
result :  [0.3651203513145447, 0.9333333373069763]
loss :  0.3651203513145447
acc :  0.9333333373069763
"""
"""
#scaler = MaxAbsScaler()
result :  [0.42169660329818726, 0.9333333373069763]
loss :  0.42169660329818726
acc :  0.9333333373069763
"""
"""
scaler = RobustScaler()
result :  [0.15321393311023712, 0.9666666388511658]
loss :  0.15321393311023712
acc :  0.9666666388511658
"""
# none, RobustScaler 승
