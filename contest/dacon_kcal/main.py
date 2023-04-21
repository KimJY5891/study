import numpy as np
from tensorflow.keras.models import Sequential #Sequential모델 
from tensorflow.keras.layers import Dense #Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리 
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
import pandas as pd
import datetime
def rmse(y_test,y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

#1. 데이터

path = "./_data/dacon_kcal/"
path_save = "./_save/dacon_kcal/"

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(7500, 10)

test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) #y가 없다. train_csv['Calories_Burned']
print(test_csv.shape) # (7500, 9)
print(train_csv.columns)
'''
Index(['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)',
       'Height(Remainder_Inches)', 'Weight(lb)', 'Weight_Status', 'Gender',
       'Age', 'Calories_Burned'],dtype='object')
'''

print(train_csv.info())
'''
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Exercise_Duration         7500 non-null   float64
 1   Body_Temperature(F)       7500 non-null   float64
 2   BPM                       7500 non-null   float64
 3   Height(Feet)              7500 non-null   float64
 4   Height(Remainder_Inches)  7500 non-null   float64
 5   Weight(lb)                7500 non-null   float64
 6   Weight_Status             7500 non-null   object
 7   Gender                    7500 non-null   object
 8   Age                       7500 non-null   int64
 9   Calories_Burned           7500 non-null   float64
'''
print(train_csv.describe())
'''
       Exercise_Duration  Body_Temperature(F)          BPM  Height(Feet)  Height(Remainder_Inches)   Weight(lb)          Age  Calories_Burned
count          7500.0000          7500.000000  7500.000000   7500.000000               7500.000000  7500.000000  7500.000000      7500.000000
mean             15.5012           104.033573    95.498133      5.248800                  5.717600   165.361187    42.636000        89.373467
std               8.3553             1.412845     9.587331      0.556663                  3.497315    33.308136    16.883188        62.817086
min               1.0000            98.800000    69.000000      4.000000                  0.000000    79.400000    20.000000         1.000000
25%               8.0000           103.300000    88.000000      5.000000                  3.000000   138.900000    28.000000        35.000000
50%              15.0000           104.400000    95.000000      5.000000                  6.000000   163.100000    39.000000        77.000000
75%              23.0000           105.100000   103.000000      6.000000                  9.000000   191.800000    56.000000       138.000000
max              30.0000           106.700000   128.000000      7.000000                 12.000000   291.000000    79.000000       300.000000
'''
print(type(train_csv))
'''
<class 'pandas.core.frame.DataFrame'>
'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의

# Gender
le.fit(train_csv['Gender']) #
aaa = le.transform(train_csv['Gender']) # 0과 1로 변화
print(aaa.shape)
print(type(aaa))
#print(np.unique(aaa,return_count=True))
train_csv['Gender'] = aaa #다시 집어넣기
print(train_csv)
test_csv['Gender'] = le.transform(test_csv['Gender'])
print(np.unique(aaa))

#Weight_Status
le.fit(train_csv['Weight_Status']) #
bbb = le.transform(train_csv['Weight_Status']) # 0과 1로 변화
print(bbb.shape)
print(type(bbb))
# print(np.unique(aaa,return_count=True))
train_csv['Weight_Status'] = bbb #다시 집어넣기
# print(train_csv)
test_csv['Weight_Status'] = le.transform(test_csv['Weight_Status'])
print(np.unique(bbb)) #[0 1 2]


####################################### 결측치 처리 #####################################
print(train_csv.isnull().sum())
# 모두 0
#####################train_csv 데이터에서 x와 y를 분리#######################

x = train_csv.drop(['Calories_Burned'],axis=1)#(1328, 9)
#drop : 빼버리겠다.엑시즈 열
#두 개이상 리스트
print("x : ", x) # [7500 rows x 9 columns]
y = train_csv['Calories_Burned']
print(y) # Name: Calories_Burned, Length: 7500, dtype: float64

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715
)
print("x_train.shape : ",x_train.shape) #(6750, 9)
print("y_train.shape : ",y_train.shape) #(6750,)
print("x_test.shape : ",x_test.shape) #(750, 9)
print("y_test.shape : ",y_test.shape) # (750,)

# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("x_train.shape01: ",x_train.shape)
print("y_train.shape01 : ",y_train.shape) 


# 2.모델구성

model = Sequential()
model.add(Dense(100,activation='relu',input_dim=9))
model.add(Dense(150,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(76,activation='relu'))
model.add(Dense(48,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3.컴파일,훈련

model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=40,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=128,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])

#4. 평가 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)

y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 스코어 : ', r2)
# rmse 만들기

rmse = rmse(y_test,y_predict) #함수 실행
print("rmse :",rmse)

y_submit = model.predict(test_csv)

#count에 넣기 
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
#print(submission)
submission['Calories_Burned'] =y_submit
submission.to_csv(path_save+date+'submit.csv')

'''
하라
test 데이터 셋에 pd.get_dummies() 함수 적용
test 데이터 셋의 결측치 처리 시 test 데이터 셋의 통계 값 활용
label encoding, one-hot encoding 시 test 데이터 셋 활용
data scaling 적용 시 test 데이터 셋 활용
'''
