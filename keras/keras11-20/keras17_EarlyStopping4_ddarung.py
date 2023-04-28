#데이콘  따릉이 문제 풀이
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split 
#대회에서 rmse로 사용 
#우리는 rmse 모르니까 유사지표 사용 
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
import pandas as pd
# 우리가 사용할 수 있도록 바꿔야함 

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


#결집치 처리해야한다.
#빨간 점 직전까지 실행 
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
#drop : 빼버리겠다.엑시즈 열
#두 개이상 리스트
print("x : ", x)

y = train_csv['count']
print(y) #(1328,)
#####################train_csv 데이터에서 x와 y를 분리#######################

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
print("x_train.shape01: ",x_train.shape) #(929, 9)
print("y_train.shape01 : ",y_train.shape) #(929, )

#2.모델구성# activation = 'linear ' 그냥 선형함수

model = Sequential()
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
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
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1600,batch_size=700,
                validation_split=0.2,
                verbose=1)

#4. 평가 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 스코어 : ', r2)

def rmse(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = rmse(y_test,y콘  따릉이 문제 풀이
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split 
#대회에서 rmse로 사용 
#우리는 rmse 모르니까 유사지표 사용 
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
import pandas as pd
# 우리가 사용할 수 있도록 바꿔야함 

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


#결집치 처리해야한다.
#빨간 점 직전까지 실행 
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
#drop : 빼버리겠다.엑시즈 열
#두 개이상 리스트
print("x : ", x)

y = train_csv['count']
print(y) #(1328,)
#####################train_csv 데이터에서 x와 y를 분리#######################

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
print("x_train.shape01: ",x_train.shape) #(929, 9)
print("y_train.shape01 : ",y_train.shape) #(929, )

#2.모델구성# activation = 'linear ' 그냥 선형함수

model = Sequential()
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
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
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1600,batch_size=700,
                validation_split=0.2,
                verbose=1)

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
submission  = pd.read_csv(path+'submission.csv',index_col=0)
submission['count'] =y_submit
submission.to_csv(path_save+'submit_0308_2009csv')
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['val_loss'],marker='.',c='red',label='val_loss') # 뭔가 명시하지 않아도 된다는데 
plt.plot(hist.history['loss'],marker='.',c='blue',label='loss') # 뭔가 명시하지 않아도 된다는데 
plt.title('따릉이') #이름 지어주기
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()
'''
타입 확인은 판다스구나~~~ 확인~~~~ 잘 됐네
판다스가 추가 삭제가 된다. 
데이터 다루기 가능 
Dense : 
model.add(Dense(2,input_dim=9))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
epoch : 1700
batch_size : 32
loss() :  2579.40673828125
r2스코어 :  0.6124584396346511
'''
'''
model = Sequential()
model.add(Dense(2,input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
epochs=800,batch_size=400
r2 스코어 :  0.6349060054370569
rmse : 51.23878272171914
loss:  2625.4130859375
'''

# 서브밋을 기준으로 잡이야 한다. 
# ...--

"""
    validation_split=0.2    
loss:  6036.880859375
5/5 [==============================] - 0s 999us/step
r2 스코어 :  0.1605020041825338
rmse : 77.69736505560977
"""
"""
   non- validation_split=0.2 
       loss:  2844.63427734375
5/5 [==============================] - 0s 994us/step
r2 스코어 :  0.6044207360819098
rmse : 53.33511281241857
"""
"""
model.add(Dense(200,input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=200,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1500,batch_size=300,
                validation_split=0.2,
                verbose=1)
                
"""
"""
   non- validation_split=0.2 
       loss:  2844.63427734375
5/5 [==============================] - 0s 994us/step
r2 스코어 :  0.6044207360819098
rmse : 53.33511281241857
"""
"""
model.add(Dense(200,input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=200,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1500,batch_size=300,
                validation_split=0.2,
                verbose=1)
                
""""""
   non- validation_split=0.2 
       loss:  2844.63427734375
5/5 [==============================] - 0s 994us/step
r2 스코어 :  0.6044207360819098
rmse : 53.33511281241857
"""
"""
model.add(Dense(200,input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1500,batch_size=300,
                validation_split=0.2,
                verbose=1)
r2 스코어 :  0.6061073571552931
rmse : 53.22128958620952                
"""
"""
model.add(Dense(200,input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=15,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1200,batch_size=300,
                validation_split=0.2,
                verbose=1)              
"""
"""
model.add(Dense(200,input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=15,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1200,batch_size=300,
                validation_split=0.2,
                verbose=1) 
                0308_2017      
                loss:  2720.28173828125
r2 스코어 :  0.6217134135705817
rmse : 52.15631865564249      
"""
"""
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=15,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1200,batch_size=300,
                validation_split=0.2,
                verbose=1) 
                0308_2020   
loss:  2648.90625
r2 스코어 :  0.6316389848715851
rmse : 51.46752564498972
"""
"""
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=12,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1200,batch_size=300,
                validation_split=0.2,
                verbose=1) 
                0308_2024
loss:  3511.544921875
5/5 [==============================] - 0s 1ms/step
r2 스코어 :  0.5116791256418005
rmse : 59.25828847163981
"""
"""
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=12,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1200,batch_size=300,
                validation_split=0.2,
                verbose=1) 
                0308_2027 
              loss:  2501.8154296875
5/5 [==============================] - 0s 997us/step
r2 스코어 :  0.6520936677516197
rmse : 50.018150127996854  

"""
"""
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=12,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1600,batch_size=300,
                validation_split=0.2,
                verbose=1) 
                0308_2032
loss:  2497.34912109375
5/5 [==============================] - 0s 1ms/step
r2 스코어 :  0.6527147245119905
rmse : 49.973485827399685

"""
"""
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=12,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 2000,batch_size=300,
                validation_split=0.2,
                verbose=1) 
                0308_2036
loss:  2984.384033203125
5/5 [==============================] - 0s 1ms/step
r2 스코어 :  0.5849869298606074
rmse : 54.62951516445038
23/23 [==============================] - 0s 1ms/step
"""
"""
model.add(Dense(200,activation = 'relu',input_dim=9))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(14))
model.add(Dense(22))
model.add(Dense(38))
model.add(Dense(48))
model.add(Dense(200,activation = 'relu'))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(76))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
               verbose=1,restore_best_weights=True)
hist =model.fit(x_train,y_train,epochs= 1600,batch_size=300,
                validation_split=0.2,
                verbose=1) 
                0308_2039
loss:  3046.0556640625
5/5 [==============================] - 0s 1ms/step
r2 스코어 :  0.576410751662835
rmse : 55.19108388044764
"""
