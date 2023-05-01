'''
머신러닝모델은 3차원이 안먹힌다. 
다차원 엑스지 부스터 못돌린다. 
dnn 개념으로 생각해볼수도 있다. 

2번째 방식 
공기가 좋은 지역 
지역을 라벨 인코딩 
시계열 데이터 
일시를 가지고 날자데이터에서 월과 시간을 빼는 것 

수상시 변수랑 주석같은 거 다 바꾸기 
'''
'''
라벨링 한 데이터 스케일링한다 안한다. ? 
라벨 인코딩 했을 경우, 
공주가 1이고 홍성이 17일이라고 할 때,
스케일링 할 필요 없다. 
카테고리 형태 데이터이기 때문에 
원핫 인코딩 해준다. 
선생님께서는 원핫 안하고 돌릴 수 있다. 
원핫 할 경우 1 -> 17개의 컬럼으로 늘어가서 데이터에 영향을 많이 미친다. 
다른 건 컬럼 1개씩이라서 17개와 쨉이 안된다. 
그럴 경우 과적합이 된다. 
pca lda 사용해볼 수도 있다. 
그냥 해보고 
원핫하고 
원핫 + 차원 축소 
0과 23시 차이는 숫자 차이는 23 이지만 사실상 1시간 차이 
1과 12시 차이는 숫자 차이는 12이지만 사실상 1달 차이 
라벨인코딩이나 원핫되는애들은 우리가 평소 사용했던 스케일러 아니고 아래 함수로 해야한다. 
sin, cos함수같은 주기 성있는 함수 라고 생각하고 
주기 함수에 넣고 수정한다. 될수도 있고 아닐 수도 있고 
'''
# 라벨 인코딩# 시계열
import numpy as np
import pandas as pd
import glob
import os
from tensorflow.keras.models import Sequential #Sequential모델 
from tensorflow.keras.layers import Dense #Dense
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
le_col_name = ['일시', '측정소']

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
# 이 파일에 든 모든 데이터를 텍스트화 하는 것
print(train_files) 
# /와\, \\,// 표현의 의미는 같다. 
# 역슬레시 \n는 줄바꿈 
# 역슬레시 \a는 띄어쓰기 
# 예약어가 있기때문에 두 개 짜리가 있다. 
test_input_files = glob.glob(path+'test_input/*.csv')
# 경로는 대소문자 상관없다. 
print(test_input_files)
'''
리스트 형태로 경로가 되어있다. 
for문형태로 read_csv 해주면 된다. 
['./_data/aifact_05/TRAIN\\공주.csv', './_data/aifact_05/TRAIN\\노은동.csv', './_data/aifact_05/TRAIN\\논산.csv', './_data/aifact_05/TRAIN\\
대천2동.csv', './_data/aifact_05/TRAIN\\독곶리.csv', './_data/aifact_05/TRAIN\\동문동.csv', './_data/aifact_05/TRAIN\\모종동.csv', './_data/aifact_05/TRAIN\\문창동.csv', './_data/aifact_05/TRAIN\\성성동.csv', './_data/aifact_05/TRAIN\\신방동.csv', './_data/aifact_05/TRAIN\\신흥동.csv', './_data/aifact_05/TRAIN\\아름동.csv', './_data/aifact_05/TRAIN\\예산군.csv', './_data/aifact_05/TRAIN\\읍내동.csv', './_data/aifact_05/TRAIN\\이원면.csv', './_data/aifact_05/TRAIN\\정림동.csv', './_data/aifact_05/TRAIN\\홍성읍.csv']
['./_data/aifact_05/test_input\\공주.csv', './_data/aifact_05/test_input\\노은동.csv', './_data/aifact_05/test_input\\논산.csv', './_data/aifact_05/test_input\\대천2동.csv', './_data/aifact_05/test_input\\독곶리.csv', './_data/aifact_05/test_input\\동문동.csv', './_data/aifact_05/test_input\\모종동.csv', './_data/aifact_05/test_input\\문창동.csv', './_data/aifact_05/test_input\\성성동.csv', './_data/aifact_05/test_input\\신방동.csv', './_data/aifact_05/test_input\\신흥동.csv', './_data/aifact_05/test_input\\아름동.csv', './_data/aifact_05/test_input\\예산군.csv', './_data/aifact_05/test_input\\읍내동.csv', './_data/aifact_05/test_input\\이원면.csv', './_data/aifact_05/test_input\\정림동.csv', './_data/aifact_05/test_input\\홍성읍.csv']
'''

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
# 테스트에 잇는 라벨 인코더로 사용해야한다. 그래야 서로 인코딩값이 같다. 
# 공주가 트레인에서 1일 대, 다르게 핏 트랜스폼해주면 공주가 0일 될 수도 있고해서 서로 똑같이해줘야하기 때문에 
# fit train만 해줘야한다. 
print(test_input_dataset) # [596088 rows x 5 columns]

train_dataset = train_dataset.drop(['측정소'],axis=1)
test_input_dataset= test_input_dataset.drop(['측정소'],axis=1)
print(test_input_dataset) # [131376 rows x 4 columns]

################### 일시 -> 월, 일, 시간으로 분리 ###############################
# 12-31 21 : 00 / 12와 21 추출 
print(train_dataset['일시'].info()) #info에서 str이라고 생각하면 쉽다. 
# date 타입 형태일경우 방법을 다르게 해야한다. 
print(type(train_dataset['일시'][0])) # <class 'str'>
print(train_dataset['일시'].dtype) # object
train_dataset['month'] = train_dataset['일시'].str[:2]
print(train_dataset['month'])
# train_dataset['day'] = train_dataset['일시'].str[3:5]
# print(train_dataset['day'])
train_dataset['hour'] = train_dataset['일시'].str[6:8]
print(train_dataset['hour'])

train_dataset = train_dataset.drop(['일시'],axis=1)

######### str -> int
# train_dataset['month'] = pd.to_numeric(train_dataset['month']) 
# to_numeric : 수치형으로 바꿔준다. 
train_dataset['month'] = train_dataset['month'].astype('Int16')
# train_dataset['day'] = train_dataset['day'].astype('Int8')
train_dataset['hour'] = train_dataset['hour'].astype('Int16')

test_input_dataset['month'] = test_input_dataset['일시'].str[:2]
# test_input_dataset['day'] = test_input_dataset['일시'].str[3:5]
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8]
test_input_dataset['month'] = test_input_dataset['month'].astype('Int16')
# test_input_dataset['day'] = test_input_dataset['day'].astype('Int8')
test_input_dataset['hour'] = test_input_dataset['hour'].astype('Int16')
test_input_dataset= test_input_dataset.drop(['일시'],axis=1)
# 장점 : 메모리에 대한 조절 가능 
# 메모리에 데이터를 넣어서 연산시키는데 int32는 32비트를 차지한다. 
# 한개의 글씨가 1비트라고 할 때 
# month의 12개의 숫자가 있는데 32 비트 사용할 필요없고 8비트로 사용해도 된다. 
# int32 보단 int8 
# 메모리 터지는 증상 방지
# 1) 배치를 줄인다. 
# 2) 데이터 타입을 바꾼다. 


# print(train_dataset.info())
'''
결측치는 못바출 부분 
일단 간단하게 처음에는 삭제해보고 나중에 모델 돌려보기 
0) 처음에는 깔끔하게 죽여버리는 것도 괜찮다. 
1) 모델 돌려서 결측치 맞추기
'''

############ 결측치 제거 PM2.5에 15542개 있다. #################
# 전체 596085-> 580546으로 줄이다
train_dataset = train_dataset.dropna()
print(train_dataset.info())
# 파생컬럼( 서로 조합해서 만든 )
# 1. 공휴일 요일  -> 공장을 더 돌릴지 안돌릴지 모르기 때문에 어떻게 될 지 ㅗㅁ른다. 
# 2. (중요) 계절(시즌) -> 

###### 시즌 - 파생피처도 생각해봐 !!!! ###### 



##### 트레인에 대한 준비 끝 위에 시즌은 알아서 만들어보기 


y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'],axis=1)
print(x,'\n',y)
# 소수점 데이터라도 스케일은 모든 데이터에 해아한다. 
# 데이터 크기 때문이 아니라 이상치 잡고 그런 의미도 있다. 
# 스케일해야 점수 잘 나올 수도 있다. 
# 부스팅은 스케일 안해 될 수도 있다. 
x_train,x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.2, random_state=337,# 랜덤 스테이트 바꾸기 
    # 지금은 엑스지 부스터로 풀거라서 아직 시계열아님 랜덤 스테이트 해도 됌
    # dnn형태로 풀고 잇다. 
    # 시계열 데이터라고 해도 x_split으로 행을 자르고 섞어버리면 된다.
    shuffle=True,
)

parameter =  {
    "n_estimators" : 300, # 디폴트 100 / 1 ~ inf / 정수
    "learning_rate" : 0.3, # 디폴트 0.3 / 0 ~ 1 / eta
    "max_depth" : 10, # 디폴트 6 / 0 ~ inf / 정수
    # "gamma" : 1, # 디폴트 0 / 0 ~ inf 
    # "min_child_weight" : 1, # 디폴트 1 / 0 ~ inf 
    # "subsample" : 3, # 디폴트 1 / 0 ~ 1 
    # "colsample_bytree" : 1, # 디폴트 / 0 ~ 1 
    # "colsample_bylevel":1, # 디폴트 / 0 ~ 1 
    # "colsample_bynode":1, # 디폴트 / 0 ~ 1 
    # "reg_alpha":0, # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
    # "reg_lambda":1, # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
    # "random_state":3377, # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
    # "verbose":0, 
    "n_jobs" : -1
}

#2. 모델 
model = XGBRegressor()

#3. 훈련 
model.set_params(**parameter,
                 eval_metric='mae',
                 early_stopping_round=200,
                 )
# model.compile에 있는 애들을 set_params에 사용한다고 생각 
# 나중에 파이토치 <-> 텐서플로우 이런식으로 바꾸는거 가능해야 한다. 
start= time.time()
model.fit(x_train,y_train,verbose=1,
          eval_set=[(x_train,y_train),(x_test,y_test)]
)

end= time.time()
print('걸린 시간 : ',round(end -start,2),'초')
# 4. 평가, 예측
y_pred = model.predict(x_test)
result = model.score(x_test,y_test) # r2 이건 스코어가 알아서 지정한것이다. 
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
submission_csv.to_csv(path_save + '0501_01.csv',encoding='utf-8')
