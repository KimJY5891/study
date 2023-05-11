# 데이터는 aws 사용 안함 

# 라벨 인코딩# 시계열
import numpy as np
import pandas as pd
import glob
import os
from tensorflow.keras.models import Sequential  #Sequential모델 
from tensorflow.keras.layers import Dense, LSTM, Bidirectional #Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import all_estimators
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pprint
import time
import warnings
warnings.filterwarnings('ignore')
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
col_name = ['연도','일시', '측정소', 'PM2.5']
col_name_after = ['연도','일시', '측정소', 'PM2.5']
le_col_name = ['일시', '측정소']
path = "./_data/aifact_05/"
path_train = "./_data/aifact_05/TRAIN/"
path_save = "./_save/aifact_05/"
path_data_imputer = "./_data/aifact_05/train_pd_inter/"
path_data_inter = "./_data/aifact_05/train_pd_inter/"
#########################################    ######################################
path = "./_data/aifact_05/"
path_train = "./_data/aifact_05/TRAIN/"
path_train_aws = "./_data/aifact_05/TRAIN_AWS/"
path_test = "./_data/aifact_05/TEST/"
path_test_aws = "./_data/aifact_05/TEST_AWS/"
path_META = "./_data/aifact_05/META/"
path_save = "./_save/aifact_05/"
path_data_imputer = "./_data/aifact_05/train_pd_inter/"
path_data_inter = "./_data/aifact_05/train_pd_inter/"

# 1. 데이터

train_files = glob.glob(path+'TRAIN/*.csv')
# print(train_files) 

test_input_files = glob.glob(path+'test_input/*.csv')
print(test_input_files)

############################# train 폴더 #########################################
train_li=[]
for filename in train_files :
    df = pd.read_csv(filename,index_col=None, 
                     header=0, # 위에 컬럼에 대한 인식하게 함 
                     encoding='utf-8')
    train_li.append(df)
    
print(train_li)
print(len(train_li)) # 리스트 안에 판다스가 17개 
# [35064 rows x 4 columns]의 판다스가 17개

train_dataset= pd.concat(train_li,axis=0,
                         ignore_index=True # 원래 잇던 인덱스는 사라지고 새로운 인덱스가 생성된다. 
                         )
print(train_dataset)
# [596088 rows x 4 columns] 합쳐진 모습 


############################# test 폴더 #########################################

test_li=[]
for filename in test_input_files :
    df = pd.read_csv(filename,index_col=None, 
                     header=0, # 위에 컬럼에 대한 인식하게 함 
                     encoding='utf-8')
    test_li.append(df)
    
print(test_li)
print(len(test_li)) # 리스트 안에 판다스가 17개 
# [35064 rows x 4 columns]의 판다스가 17개

test_input_dataset= pd.concat(test_li,axis=0,
                         ignore_index=True # 원래 잇던 인덱스는 사라지고 새로운 인덱스가 생성된다. 
                         )
print(test_input_dataset)
# [596088 rows x 4 columns] 합쳐진 모습 

############################# 라벨 인코더 #########################################

le=LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])
print(test_input_dataset) # [596088 rows x 5 columns]

train_dataset = train_dataset.drop(['측정소'],axis=1)
test_input_dataset= test_input_dataset.drop(['측정소'],axis=1)
print(test_input_dataset) # [131376 rows x 4 columns]

################### 일시 -> 월, 일, 시간으로 분리 ###############################
# 12-31 21 : 00 / 12와 21 추출 
print(type(train_dataset['일시'][0])) # <class 'str'>
print(train_dataset['일시'].dtype) # object
train_dataset['month'] = train_dataset['일시'].str[:2]
print(train_dataset['month'])
train_dataset['day'] = train_dataset['일시'].str[3:5]
# print(train_dataset['day'])
train_dataset['hour'] = train_dataset['일시'].str[6:8]
print(train_dataset['hour'])

train_dataset = train_dataset.drop(['일시'],axis=1)

######### str -> int
# train_dataset['month'] = pd.to_numeric(train_dataset['month']) 
# to_numeric : 수치형으로 바꿔준다. 
train_dataset['month'] = train_dataset['month'].astype('float')
train_dataset['day'] = train_dataset['day'].astype('float')
train_dataset['hour'] = train_dataset['hour'].astype('float')

test_input_dataset['month'] = test_input_dataset['일시'].str[:2]
test_input_dataset['day'] = test_input_dataset['일시'].str[3:5]
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8]
test_input_dataset['month'] = test_input_dataset['month'].astype('float')
test_input_dataset['day'] = test_input_dataset['day'].astype('float')
test_input_dataset['hour'] = test_input_dataset['hour'].astype('float')
test_input_dataset= test_input_dataset.drop(['일시'],axis=1)

############ 결측치 제거 PM2.5에 15542개 있다. #################
# 전체 596085-> 580546으로 줄이다
train_dataset = train_dataset.interpolate(order=1)

# 파생컬럼( 서로 조합해서 만든 )
# 1. 공휴일 요일  -> 공장을 더 돌릴지 안돌릴지 모르기 때문에 어떻게 될 지 ㅗㅁ른다. 
# 2. (중요) 계절(시즌) -> 

###### 시즌 - 파생피처도 생각해봐 !!!! ###### 



##### 트레인에 대한 준비 끝 위에 시즌은 알아서 만들어보기 


y =  np.array(train_dataset['PM2.5'])
x = np.array(train_dataset.drop(['PM2.5'],axis=1))
print(x,'\n',y)

x_train,x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.2, # random_state=337,
    # shuffle=True
)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)
# print(x_train.shape,x_test.shape)
# print('x_train : ',x_train)
# print('x_train[day] : ',x_train['day'])
# print('x_train[hour] : ',x_train['hour'])
# print('x_test : ',x_test)
# print(y_train.shape,y_test.shape)

#2. 모델 

start= time.time()

model= Sequential()
model.add(LSTM(64,return_sequences=True,input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(32))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 훈련 
model.compile(optimizer='adam', loss='mse')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss',mode='min',restore_best_weights=True,patience=8)
model.fit(x_test,y_test,epochs=48,verbose=1,callbacks =[es])
end= time.time()
print('걸린 시간 : ',round(end -start,2),'초')

# 4. 평가, 예측
result = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
print("model.score : ",result)
r2 = r2_score(y_test,y_pred)
print('r2 : ',r2)
mae = mean_absolute_error(y_test,y_pred)
print('mae : ',mae)

true_test = test_input_dataset[test_input_dataset['PM2.5'].isnull()].drop('PM2.5',axis=1)

# 제출 
submission_csv = pd.read_csv(path +'answer_sample.csv')
y_submit = model.predict(true_test)
submission_csv['PM2.5'] = y_submit
submission_csv.to_csv(path_save + '0503_01.csv',encoding='utf-8')
