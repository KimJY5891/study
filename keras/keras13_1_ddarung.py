
#데이콘  따릉이 문제 풀이
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential #Sequential모델 
from tensorflow.keras.layers import Dense #Dense
from sklearn.model_selection import train_test_split 
#대회에서 rmse로 사용 
#우리는 rmse 모르니까 유사지표 사용 
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
import pandas as pd
# 우리가 사용할 수 있도록 바꿔야함 

#1. 데이터

path = "./_data/ddarung/"
#./: 현재폴더 | 여기서 .는 스터디폴더

train_csv=pd.read_csv(path + 'train.csv',index_col=0) #index_col은 인덱스 컬럼이 뭐냐? #인덱스는 데이터가 아니니까 빼야지! 
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
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
#

#결집치 처리해야한다.
#빨간 점 직전까지 실행 
print(train_csv.describe())
#min: 최소값 ,max : 최대값, 50%: 중위값 
print(type(train_csv))
####################################### 결측치 처리 #######################################                                                                                   #
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
    train_size=0.7,random_state=8517
)
print("x_train.shape01: ",x_train.shape) #(929, 9)
print("y_train.shape01 : ",y_train.shape) #(929, )

#2.모델구성

model = Sequential()
model.add(Dense(24,input_dim=9))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(92))
model.add(Dense(80))
model.add(Dense(48))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1500,batch_size=150,verbose=1)

#4. 평가 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)
# nan = 데이터가 없다, 원본 데이터 값이 없어서 nan이 나온다. 0은 데이터가 있는 것 
# 결집치 값 처리 첫 번째 걍 0으로 처리



'''
Dense : 
epoch : 
batch_size : 
loss() :  
r2스코어 :  
'''
