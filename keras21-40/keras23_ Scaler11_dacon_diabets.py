import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
#1. 데이터

path = "./_data/dacon_diabets/"
path_save = "./_save/dacon_diabets/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################                                                                                 #
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)
####################################### 결측치 처리 #######################################                                                                                 #
x = train_csv.drop(['Outcome'],axis=1)#
y = train_csv['Outcome']
print("x.shape : ",x.shape)#(652, 8)
print("y.shape : ",y.shape)#(652, )
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=1234
)
# scaler = StandardScaler()
# #scaler = MinMaxScaler()
# #scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test) 
# test_csv = scaler.transform(test_csv)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(24,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(200,activation="sigmoid"))
model.add(Dense(400,activation="relu"))
model.add(Dense(1600,activation="relu"))
model.add(Dense(80,activation="relu"))
model.add(Dense(40,activation="linear"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="relu"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="linear"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',patience=400,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=3000,batch_size=1000,verbose=1,validation_split=0.2,callbacks=[es])
# 회귀에서는 매트릭에서 입력해서 볼 수 있다.
# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

y_submit = np.round(model.predict(test_csv))
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['Outcome'] =y_submit
submission.to_csv(path_save+'submit_0314_02.csv')

"""
NON

"""
"""
scaler = StandardScaler()
result :  [1.773333191871643, 0.7878788113594055, 0.1592719852924347]
acc :  0.7878787878787878
"""
"""
#scaler = MinMaxScaler()
result :  [0.4428599178791046, 0.7878788113594055, 0.14341053366661072]
acc :  0.7878787878787878
"""
"""
#scaler = MaxAbsScaler()
result :  [0.40212759375572205, 0.8484848737716675, 0.12452050298452377]
acc :  0.8484848484848485
"""
"""
scaler = RobustScaler()
result :  [0.7100448608398438, 0.8030303120613098, 0.1609112173318863]
acc :  0.803030303030303
"""
"""
대회의 test.csv는 x_pred에 해당
y가 없다. 
test.csv에서 낸 값을 
"""
