import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.metrics import r2_score, accuracy_score,mean_squared_error, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리




#1. 데이터


path = "./_data/dacon_phone/"
path_save = "./_save/dacon_phone/"
train_csv=pd.read_csv(path+'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #((30200, 13)
test_csv=pd.read_csv(path+'test.csv',index_col=0)
print(test_csv) #y가 없다.
print(test_csv.shape) #(12943, 12)
#===================================================================================================================
print(train_csv.columns)
# Index(['가입일', '음성사서함이용', '주간통화시간', '주간통화횟수', '주간통화요금', '저녁통화시간', '저녁통화횟수',
#        '저녁통화요금', '밤통화시간', '밤통화횟수', '밤통화요금', '상담전화건수', '전화해지여부'],
#       dtype='object')
print(train_csv.info())
print(train_csv.describe())
print(type(train_csv))


####################################### 결측치 처리 #######################################                                                                                 #
#결측치 처리 1. 제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)
####################################### 결측치 처리 #######################################                                                                                 #


#####################train_csv 데이터에서 x와 y를 분리#######################
x = train_csv.drop(['전화해지여부'],axis=1)#(1328, 9)
print("x : ", x.shape) #(30200, 12)
y = train_csv['전화해지여부']
print(y.shape) #(30200,)
print(np.unique(y)) #[0 1]
#####################train_csv 데이터에서 x와 y를 분리#######################


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
print("x_train : ",x_train.shape) # x_train :  (27180, 12)
print("y_train : ",y_train.shape) # y_train :  (27180,)
# scaler = RobustScaler()
# x_train = x_train.reshape(-1,1)
# x_train = scaler.fit_transform(x_train)
# x_test = x_test.reshape(-1,1)
# x_test = scaler.transform(x_test)
x_train = x_train.reshape(30200, 12, 1)
x_test = x_test.reshape(30200, 12, 1)
print("x_train : ",x_train.shape) # x_train :  (27180, 12)
print("y_train : ",y_train.shape) # y_train :  (27180,)


#2. 모델 구성


model=Sequential()
model.add(Conv1D(10,2,padding='same',input_shape=(12,1)))
model.add(Conv1D(100,2,padding='same'))
model.add(Conv1D(10,2,padding='same'))
model.add(Conv1D(10,2,padding='same'))
model.add(Conv1D(10,2,padding='same'))
model.add(Conv1D(10,2,padding='same'))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(4))
model.add(Dense(1,activation='sigmoid'))


#3. 컴파일, 훈련
# def f1_metric(y_test, y_predict):
#     return f1_score(y_test, y_predict, average='macro')
model.compile(loss='binary_crossentropy',optimizer='adam',)
              #metrics=['f1_metric'])
import time
start = time.time()
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss',
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True
)
mcp=ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                    filepath='./_save/MCP/keras_phone_mcp.hdf5') #가중치 저장
model.fit(
    x_train,y_train,
    epochs=10,batch_size=1000,
    validation_split=0.2,callbacks=[es,mcp]
)
end = time.time()


#4. 평가, 예측


result =model.evaluate(x_test,y_test)
print('result : ',result )


y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc 스코어 : ', acc)
print('time : ',round(end-start,2))
# f1 = f1_score(
#     y_test,
#     y_predict,
#     labels=None,
#     pos_label=1,
#     average='macro',
#     sample_weight=None,
#     zero_division='warn'
# )
# print('f1_score : ',f1)
print('y_predict: ',y_predict)
print('y_predict.shape: ',y_predict.shape)#(3020, 1)


# #count에 넣기
y_submit = np.round(model.predict(test_csv))
print('y_submit: ',y_submit)#
print('y_submit.shape: ',y_submit.shape)#
y_submit = y_submit.reshape(y_submit.shape[0],)
print('y_submit: ',y_submit)#
print('y_submit.shape: ',y_submit.shape)#


# submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
# submission['전화해지여부'] =y_submit
# submission.to_csv(path_save+'submit_0324.csv')
