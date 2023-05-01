#데이콘  따릉이 문제 풀이
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, S¹tandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from sklearn.metrics import r2_score, accuracy_score

from sklearn.metrics import mean_squared_error

#1. 데이터

path = "./_data/ddarung/"
path_save = "./_save/ddarung/"

#./: 현재폴더 | 여기서 .는 스터디폴더

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
#불러올 때 read_csv()
#index_col은 인덱스 컬럼이 뭐냐? #인덱스는 데이터가 아니니까 빼야지! 
#컬럼이름(header)도 데이터는 아니다. #index, header는 연산하지 않는다.
#인덱스 = 아이디.색인는 데이터가 아니고 번호 매긴것

print(train_csv)
print(train_csv.shape) #(1459, 10)
# 원래라면 이렇게 하는것 하지만 path 변수로 인해서 ㄱㅊ train_csv=pd.read_csv("./_data/ddarung/train.csv")

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
#min: 최소값 ,max : 최대값, 50%: 중위값 
print(type(train_csv))
####################################### 결측치 처리 #######################################                                                                                 #
#결측치 처리 1. 제거
#print(train_csv.isnull().sum()) # 결측치가 몇개 있는지 보여줌
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# dropna : 결측치를 삭제하겠다.
# 변수로 저장하기
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)

#####################train_csv 데이터에서 x와 y를 분리#######################

x = train_csv.drop(['count'],axis=1)#(1328, 9)
print("x : ", x)
y = train_csv['count']
print(y) #(1328,)
#####################train_csv 데이터에서 x와 y를 분리#######################

x_train, x_test, y_train, y_test = train_test_split(
    x,y,9
    train_size=0.9,random_state=8715
)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler = MaxAbsScaler()
#scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv)
print("x_train.shape01: ",x_train.shape) #(929, 9)
print("y_train.shape01 : ",y_train.shape) #(929, )

#2.모델구성# activation = 'linear ' 그냥 선형함수

model = Sequential()
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=40,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1600,batch_size=700,
                validation_split=0.2,verbose=1,callbacks=[es])

#4. 평가 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 스코어 : ', r2)

def rmse(y_test,y_predict) :  
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = rmse(y_test,y_predict) 
print("rmse :",rmse)

y_submit = model.predict(test_csv)
#y_submit = scaler.inverse_transform(y_submit) # 전처리 이전으로 되돌리기
submission  = pd.read_csv(path+'submission.csv',index_col=0)
submission['count'] =y_submit
submission.to_csv(path_save+'submit_0313_2029.csv')


"""
NON
loss:  3019.026123046875
r2 스코어 :  0.5801695676323988
"""
"""
scaler = StandardScaler()
loss:  3405.4150390625
r2 스코어 :  0.5264376940007826
rmse : 58.355933722454814
"""
"""
#scaler = MinMaxScaler()
loss:  2749.564697265625
r2 스코어 :  0.6176412779912877
rmse : 52.43629036621396
"""
"""
#scaler = MaxAbsScaler()
loss:  2099.36962890625
r2 스코어 :  0.7080584011608153
"""
"""
scaler = RobustScaler()
loss:  2880.802001953125
r2 스코어 :  0.5993911642645371
rmse : 53.67310510766492
"""
# MaxAbsScaler
"""
  #데이콘  따릉이 문제 풀이
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from sklearn.metrics import r2_score, accuracy_score

from sklearn.metrics import mean_squared_error

#1. 데이터

path = "./_data/ddarung/"
path_save = "./_save/ddarung/"

#./: 현재폴더 | 여기서 .는 스터디폴더

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
#불러올 때 read_csv()
#index_col은 인덱스 컬럼이 뭐냐? #인덱스는 데이터가 아니니까 빼야지! 
#컬럼이름(header)도 데이터는 아니다. #index, header는 연산하지 않는다.
#인덱스 = 아이디.색인는 데이터가 아니고 번호 매긴것

print(train_csv)
print(train_csv.shape) #(1459, 10)
# 원래라면 이렇게 하는것 하지만 path 변수로 인해서 ㄱㅊ train_csv=pd.read_csv("./_data/ddarung/train.csv")

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
#min: 최소값 ,max : 최대값, 50%: 중위값 
print(type(train_csv))
####################################### 결측치 처리 #######################################                                                                                 #
#결측치 처리 1. 제거
#print(train_csv.isnull().sum()) # 결측치가 몇개 있는지 보여줌
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# dropna : 결측치를 삭제하겠다.
# 변수로 저장하기
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)

#####################train_csv 데이터에서 x와 y를 분리#######################

x = train_csv.drop(['count'],axis=1)#(1328, 9)
print("x : ", x)
y = train_csv['count']
print(y) #(1328,)
#####################train_csv 데이터에서 x와 y를 분리#######################

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler = MaxAbsScaler()
#scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv)
print("x_train.shape01: ",x_train.shape) #(929, 9)
print("y_train.shape01 : ",y_train.shape) #(929, )

#2.모델구성# activation = 'linear ' 그냥 선형함수

model = Sequential()
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1600,batch_size=700,
                validation_split=0.2,verbose=1,callbacks=[es])

#4. 평가 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 스코어 : ', r2)

def rmse(y_test,y_predict) :  
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = rmse(y_test,y_predict) 
print("rmse :",rmse)

y_submit = model.predict(test_csv)
#y_submit = scaler.inverse_transform(y_submit) # 전처리 이전으로 되돌리기
submission  = pd.read_csv(path+'submission.csv',index_col=0)
submission['count'] =y_submit
submission.to_csv(path_save+'submit_0313_2026.csv')  
"""
