#캐글 
from sklearn.metrics import r2_score, mean_squared_log_error
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
# 여러 데이터를 읽기 쉽다.

# 1. 데이터
path = "./_data/kaggle/"
path_save = "./_save/kaggle/"
train_csv = pd.read_csv(path + 'train.csv',index_col=0)
print("train_csv : ",train_csv)
print("train_csv.shape : ", train_csv.shape)

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
print("test_csv : ",test_csv)
print("test_csv.shape : ", test_csv.shape)
print("train_csv.columns : ",train_csv.columns) #test_csv.shape :  (6493, 8)
# train_csv.columns :  Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
print('train_csv.info() : ',train_csv.info()) #train_csv.info() :  None
print('type(train_csv) : ',type(train_csv)) #type(train_csv) :  <class 'pandas.core.frame.DataFrame'>
#dtype: int64 (int= 정수)
#결측치 처리
print("isnull sum01 : ",train_csv.isnull().sum())
print(train_csv.shape)#(10886, 11)

train_csv = train_csv.dropna()
print('train_csv.info()02 :', train_csv.info())#None
print("isnull sum 02: ",train_csv.isnull().sum())
'''
isnull sum01 :  season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0
dtype: int64
<class 'pandas.core.frame.DataFrame'>
Index: 10886 entries, 2011-01-01 00:00:00 to 2012-12-19 23:00:00
Data columns (total 11 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   season      10886 non-null  int64
 1   holiday     10886 non-null  int64
 2   workingday  10886 non-null  int64
 3   weather     10886 non-null  int64
 4   temp        10886 non-null  float64
 5   atemp       10886 non-null  float64
 6   humidity    10886 non-null  int64
 7   windspeed   10886 non-null  float64
 8   casual      10886 non-null  int64
 9   registered  10886 non-null  int64
 10  count       10886 non-null  int64
dtypes: float64(3), int64(8)
memory usage: 1020.6+ KB
train_csv.info()02 : None
isnull sum 02:  season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0
dtype: int64
'''
print(train_csv.shape)#(10886, 11)

# x와 y를 분리
x= train_csv.drop(['count'],axis=1) 
print("x : ",x)
print("x.shape : ",x.shape) #(10886, 10)
y=train_csv['count']
print("y : ",y)
print("y.shape : ",y.shape) #(10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7, random_state= 8715
)
print("x_train.shape : ",x_train.shape ) #(7620, 10)
print("y_train.shape : ",y_train.shape ) #(7620,)

# 2. 모델구성
model=Sequential()
model.add(Dense(2,input_dim=10))
model.add(Dense(4,activation='relu'))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(32))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=900,batch_size=100,verbose=1)

# 4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss : ',loss)
"""
마이너스가 나오는 이유
처음에 랜덤하게 선을 긋고 시작해서 
활성화 함수
한정화 함수 
나중에 다룸 
relu  0이상의  값은 양수로 유지 0이하의 값은 0이 되는 함수 
output dim 히든 레이어에 넣는다.
activation = 'ralu'
 
"""
"""
model=Sequential()
model.add(Dense(2,input_dim=10))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(32))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
epochs=600,batch_size=100,verbose=1
77/77 [==============================] - 0s 2ms/step - loss: 0.0013
103/103 [==============================] - 0s 869us/step - loss: 0.0023
loss :  0.00228860042989254
train_size=0.7, random_state= 8715
"""
"""
model=Sequential()
model.add(Dense(2,input_dim=10))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(32))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
epochs=800,batch_size=100,verbose=1
77/77 [==============================] - 0s 2ms/step - loss: 0.0013
103/103 [==============================] - 0s 869us/step - loss: 0.0023
loss :  6.70817171339877e-05
train_size=0.7, random_state= 8715
"""
    
    
