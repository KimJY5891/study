import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.metrics import r2_score, accuracy_score,mean_squared_error


#1. 데이터


path = "./_data/dacon_ddraung/"
path_save = "./_save/dacon_ddraung/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(1459, 10)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) #y가 없다.
print(test_csv.shape) #((715, 9) #count 값이 없다.
#===================================================================================================================
print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
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
#####################train_csv 데이터에서 x와 y를 분리#######################
x = train_csv.drop(['count'],axis=1)#(1328, 9)
print("x : ", x)
y = train_csv['count']
print(y) #(1328,)
#####################train_csv 데이터에서 x와 y를 분리#######################


x=x.values.reshape(1328, 9, 1)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
print("x_train.shape01: ",x_train.shape) #(929, 9)
print("y_train.shape01 : ",y_train.shape) #(929, )


#2.모델구성


Input01 = Input(shape=(9,1))
Conv1D01 = Conv1D(10,2,padding='same')(Input01)
Conv1D02 = Conv1D(10,2,padding='same')(Conv1D01)
Flatten01 = Flatten()(Conv1D02)
Dense01 = Dense(12)(Flatten01)
Dense02 = Dense(12)(Dense01)
Output01 = Dense(1)(Dense02)
model = Model(inputs=Input01,outputs=Output01)




#3.컴파일,훈련


model.compile(loss='mse',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='acc',patience=10,mode='max',
                   verbose=1,restore_best_weights=True)
mcp =ModelCheckpoint(monitor='acc',mode='max',verbose=1,save_best_only=True,
                     filepath='./_save/MCP/keras48_Conv1D_04_dacon_ddaraung_mcp.hdf5')
model.fit(x_train,y_train,epochs=800,batch_size=400,verbose=1,callbacks=[es,mcp])




#4. 평가 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)


y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 스코어 : ', r2)
# rmse 만들기
def rmse(y_test,y_predict) :  #함수 정의만 한 것
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = rmse(y_test,y_predict) #함수 실행
print("rmse :",rmse)


#count에 넣기
y_submit = model.predict(test_csv)
submission  = pd.read_csv(path+'submission.csv',index_col=0)
submission['count'] =y_submit
submission.to_csv(path_save+'submit_0307_1630.csv')
