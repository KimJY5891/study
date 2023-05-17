import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.utils import all_estimators
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
def split_x(dataset, timesteps): # split_x라는 함수를 정의
    aaa = [] # aaa 에 빈칸의 리스트를 만든다.
    for i in range(len(dataset) - timesteps + 1): # for : 반복문, i 변수에, range 일정 간격으로 숫자를 나열 len 데이터의 길이 시작값 - 끝값 + 증가값, in 과 : 사이가 반복할 횟수 
        subset = dataset[i : (i + timesteps)] # subset 변수에 dataset i0 : (i0 + 5)
        aaa.append(subset) # aaa 리스트에 subset 값 이어붙힌다. aaa.i0 : (i0 + 5)
    return np.array(aaa) # 충족할때까지 반복한다.
import pprint
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
col_name = ['연도','일시', '측정소', 'PM2.5']
le_col_name = ['일시', '측정소']
path = "./_data/aifact_05/"
path_train = "./_data/aifact_05/TRAIN/"
path_test = "./_data/aifact_05/TEST_INPUT/"
path_train_AWS = "./_data/aifact_05/TRAIN_AWS/"
path_test_AWS = "./_data/aifact_05/TEST_AWS/"
path_save = "./_save/aifact_05/"
path_data_imputer = "./_data/aifact_05/train_pd_inter/"
path_data_inter = "./_data/aifact_05/train_pd_inter/"
data_name_list = [
    '공주.csv','노은동.csv','논산.csv','대천2동.csv','독곶리.csv','동문동.csv',
    '모종동.csv','문창동.csv','성성동.csv','신방동.csv','신흥동.csv','아름동.csv','예산군.csv',
    '읍내동.csv','이원면.csv','정림동.csv','홍성읍.csv']
path_meta = './_data/aifact_05/META/'


#### 1. 데이터
# 예측을 위한 데이터 불러오기
# pm
gongjoo_train_csv=pd.read_csv(path_train +'공주.csv',encoding='utf-8',index_col=False)
# aws
gongjoo_train_AWS_csv=pd.read_csv(path_train_AWS +'공주.csv',encoding='utf-8',index_col=False)
jungan_train_AWS_csv=pd.read_csv(path_train_AWS +'정안.csv',encoding='utf-8',index_col=False)
segold_train_AWS_csv=pd.read_csv(path_train_AWS +'세종금남.csv',encoding='utf-8',index_col=False)

bogan_list  = [gongjoo_train_csv,gongjoo_train_AWS_csv,jungan_train_AWS_csv, segold_train_AWS_csv]

# sub
gongjoo_test_AWS_csv=pd.read_csv(path_test_AWS +'공주.csv',encoding='utf-8',index_col=False)
jungan_test_AWS_csv=pd.read_csv(path_test_AWS +'정안.csv',encoding='utf-8',index_col=False)
segold_test_AWS_csv=pd.read_csv(path_test_AWS +'세종금남.csv',encoding='utf-8',index_col=False)
gongjoo_test_input_csv=pd.read_csv( path_test +'공주.csv',encoding='utf-8',index_col=False)

# 결측치 
nan_index = gongjoo_test_input_csv.isna().any(axis=1)
print(nan_index)
# boolean 형태의 시리즈를 사용하여 nan 값을 가지는 행의 인덱스를 추출합니다.
nan_rows = gongjoo_test_input_csv.index[nan_index]
nan_rows=nan_rows.astype('int16').to_list()
lst = [1,2,3,4,5,6,7]
print(len(lst)) # 7
def find_consecutive(lst):
    result = []
    i = 0
    while i < len(lst): # while 조건 만족할 때까지 무한 반복 즉, lst 길이보다 
        start = lst[i]
        end = start
        while i+1 < len(lst) and lst[i+1] == end+1:  
            end = lst[i+1]
            i += 1
            # i+1 < len(lst): 이 조건은 lst[i+1]에 액세스하는 동안 목록의 범위를 초과하지 않도록 보장합니다.
            # 인덱스 i+1이 목록 list 길이보다 작은지 확인합니다. 
            # i+1이 len(lst)보다 크거나 같으면 목록의 끝에 도달했으며 더 이상 확인할 연속 값이 없음을 의미합니다.
        i += 1
        if end - start > 0: # end와 start가 같다면 0, 아닐 경우 1 
            result.append((start, end))
    return result
result = find_consecutive(nan_rows)
print(result)

# 결측치 처리 
gongjoo_train_csv = gongjoo_train_csv.interpolate(order=3)
gongjoo_train_AWS_csv = gongjoo_train_AWS_csv.interpolate(order=3)
jungan_train_AWS_csv = jungan_train_AWS_csv.interpolate(order=3)
segold_train_AWS_csv = segold_train_AWS_csv.interpolate(order=3)
jungan_test_AWS_csv = jungan_test_AWS_csv.interpolate(order=3)
segold_test_AWS_csv = segold_test_AWS_csv.interpolate(order=3)
gongjoo_test_AWS_csv = gongjoo_test_AWS_csv.interpolate(order=3)

# 라벨 인코더
le=LabelEncoder()
gongjoo_train_csv['Location'] = 0
gongjoo_train_AWS_csv['Location'] = 1
jungan_train_AWS_csv['Location'] =2
segold_train_AWS_csv['Location'] = 3
gongjoo_test_AWS_csv['Location'] = 1
jungan_test_AWS_csv['Location'] = 2
segold_test_AWS_csv['Location'] = 3
gongjoo_test_input_csv['Location'] = 0

gongjoo_train_csv = gongjoo_train_csv.drop(['측정소'],axis=1)
gongjoo_train_AWS_csv = gongjoo_train_AWS_csv.drop(['지점'],axis=1)
jungan_train_AWS_csv = jungan_train_AWS_csv.drop(['지점'],axis=1)
segold_train_AWS_csv = segold_train_AWS_csv.drop(['지점'],axis=1)
gongjoo_test_AWS_csv = gongjoo_test_AWS_csv.drop(['지점'],axis=1)
jungan_test_AWS_csv = jungan_test_AWS_csv.drop(['지점'],axis=1)
segold_test_AWS_csv = segold_test_AWS_csv.drop(['지점'],axis=1)
gongjoo_test_input_csv = gongjoo_test_input_csv.drop(['측정소'],axis=1)

# 일시 변경

gongjoo_train_csv['month'] = gongjoo_train_csv['일시'].str[:2]
gongjoo_train_AWS_csv['month'] = gongjoo_train_AWS_csv['일시'].str[:2]
jungan_train_AWS_csv['month'] =jungan_train_AWS_csv['일시'].str[:2]
segold_train_AWS_csv['month'] = segold_train_AWS_csv['일시'].str[:2]
gongjoo_test_AWS_csv['month'] = gongjoo_test_AWS_csv['일시'].str[:2]
jungan_test_AWS_csv['month'] = jungan_test_AWS_csv['일시'].str[:2]
segold_test_AWS_csv['month'] = segold_test_AWS_csv['일시'].str[:2]
gongjoo_test_input_csv['month'] = gongjoo_test_input_csv['일시'].str[:2]

gongjoo_train_csv['hour'] = gongjoo_train_csv['일시'].str[6:8]
gongjoo_train_AWS_csv['hour'] = gongjoo_train_AWS_csv['일시'].str[6:8]
jungan_train_AWS_csv['hour'] =jungan_train_AWS_csv['일시'].str[6:8]
segold_train_AWS_csv['hour'] = segold_train_AWS_csv['일시'].str[6:8]
gongjoo_test_AWS_csv['hour'] = gongjoo_test_AWS_csv['일시'].str[6:8]
jungan_test_AWS_csv['hour'] = jungan_test_AWS_csv['일시'].str[6:8]
segold_test_AWS_csv['hour'] = segold_test_AWS_csv['일시'].str[6:8]
gongjoo_test_input_csv['hour'] = gongjoo_test_input_csv['일시'].str[6:8]

gongjoo_train_csv = gongjoo_train_csv.drop(['일시'],axis=1)
gongjoo_train_AWS_csv = gongjoo_train_AWS_csv.drop(['일시'],axis=1)
jungan_train_AWS_csv = jungan_train_AWS_csv.drop(['일시'],axis=1)
segold_train_AWS_csv = segold_train_AWS_csv.drop(['일시'],axis=1)
gongjoo_test_AWS_csv = gongjoo_test_AWS_csv.drop(['일시'],axis=1)
segold_test_AWS_csv = segold_test_AWS_csv.drop(['일시'],axis=1)
jungan_test_AWS_csv = jungan_test_AWS_csv.drop(['일시'],axis=1)
gongjoo_test_input_csv = gongjoo_test_input_csv.drop(['일시'],axis=1)

jungan_train_AWS_csv = jungan_train_AWS_csv.drop(['month','hour'],axis=1)
segold_train_AWS_csv = segold_train_AWS_csv.drop(['month','hour'],axis=1)

# x,y 만들기

y = gongjoo_train_csv['PM2.5']
# 'aws전부'
# x = pd.concat([gongjoo_train_AWS_csv, jungan_train_AWS_csv,segold_train_AWS_csv], axis=1)
x = gongjoo_train_AWS_csv

'''
   연도    기온(°C)   풍향(deg)   풍속(m/s)  강수량(mm)  습도(%)  Location month hour  연도    기온(°C)   풍향(deg)   풍속(m/s)  강수량(mm)  습도(%)  Location  연도    기온(°C)   풍향(deg)   풍속(m/s) 
 강수량(mm)  습도(%)  Location
0   4  0.244866  0.123333  0.038363      0.0  0.647         1    01   00   4  0.211690  0.385278  0.020460      0.0  0.720         2   4  0.244866  0.024167  0.030691      0.0  0.587         3
1   4  0.232227  0.167778  0.033248      0.0  0.648         1    01   01   4  0.195893  0.000000  0.005115      0.0  0.771         2   4  0.222749  0.000000  0.005115      0.0  0.629         3
2   4  0.206951  0.000000  0.002558      0.0  0.734         1    01   02   4  0.189573  0.000000  0.007673      0.0  0.783         2   4  0.206951  0.543333  0.023018      0.0  0.660         3
3   4  0.199052  0.000000  0.002558      0.0  0.753         1    01   03   4  0.187994  0.640833  0.012788      0.0  0.788         2   4  0.191153  0.445833  0.012788      0.0  0.725         3
4   4  0.189573  0.000000  0.002558      0.0  0.795         1    01   04   4  0.175355  0.000000  0.002558      0.0  0.832         2   4  0.183254  0.000000  0.002558      0.0  0.763         3
'''
print(x.shape) # (7728, 23)
print(y.shape) # (35064,)
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.1,
    # random_state=337,
    shuffle=False
)

print(x_train.shape,y_train.shape)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 

model = KNeighborsRegressor(
    n_neighbors=10,
    weights = 'distance'
    )

# 3. 훈련

model.fit(x_train,y_train)

# 4. 평가, 예측

y_pred = model.predict(x_test)
result = model.score(x_test,y_test) # r2 이건 스코어가 알아서 지정한것이다. 
print("model.score : ",result)

r2 = r2_score(y_test,y_pred)
print('r2 : ',r2)

mae = mean_absolute_error(y_test,y_pred)
print('mae : ',mae)

# gongjoo_test_AWS_csv = gongjoo_test_AWS_csv.drop(['month','hour'],axis=1)
jungan_test_AWS_csv = jungan_test_AWS_csv.drop(['month','hour'],axis=1)
segold_test_AWS_csv = segold_test_AWS_csv.drop(['month','hour'],axis=1)

true_test = pd.concat([gongjoo_test_AWS_csv,jungan_test_AWS_csv,segold_test_AWS_csv],axis=1)
true_test = gongjoo_test_AWS_csv
# 슬라이싱을 통해 여러 개의 행 선택

    
   
true_test = pd.concat([true_test.iloc[48:120], 
                       true_test.iloc[168:240], 
                       true_test.iloc[288:360], 
                       true_test.iloc[408:480], 
                       true_test.iloc[528:600], 
                       true_test.iloc[648:720], 
                       true_test.iloc[768:840], 
                       true_test.iloc[888:960], 
                       true_test.iloc[1008:1080], 
                       true_test.iloc[1128:1200], 
                       true_test.iloc[1248:1320], 
                       true_test.iloc[1368:1440], 
                       true_test.iloc[1488:1560], 
                       true_test.iloc[1608:1680], 
                       true_test.iloc[1728:1800], 
                       true_test.iloc[1848:1920], 
                       true_test.iloc[1968:2040], 
                       true_test.iloc[2088:2160], 
                       true_test.iloc[2208:2280], 
                       true_test.iloc[2328:2400], 
                       true_test.iloc[2448:2520], 
                       true_test.iloc[2568:2640], 
                       true_test.iloc[2688:2760], 
                       true_test.iloc[2808:2880], 
                       true_test.iloc[2928:3000], 
                       true_test.iloc[3048:3120], 
                       true_test.iloc[3168:3240], 
                       true_test.iloc[3288:3360], 
                       true_test.iloc[3408: 3480], 
                       true_test.iloc[3528: 3600], 
                       true_test.iloc[3648: 3720], 
                       true_test.iloc[3768: 3840], 
                       true_test.iloc[3888: 3960], 
                       true_test.iloc[4008: 4080], 
                       true_test.iloc[4128: 4200], 
                       true_test.iloc[4248: 4320], 
                       true_test.iloc[4248: 4320], 
                       true_test.iloc[4368: 4440], 
                       true_test.iloc[4488: 4560], 
                       true_test.iloc[4608: 4680], 
                       true_test.iloc[4728: 4800], 
                       true_test.iloc[4848: 4920], 
                       true_test.iloc[4968: 5040], 
                       true_test.iloc[5088: 5160], 
                       true_test.iloc[5208: 5280], 
                       true_test.iloc[5328: 5400], 
                       true_test.iloc[5448: 5520], 
                       true_test.iloc[5568: 5640], 
                       true_test.iloc[5688: 5760], 
                       true_test.iloc[5808: 5880], 
                       true_test.iloc[5928: 6000], 
                       true_test.iloc[6048: 6120], 
                       true_test.iloc[6168: 6240], 
                       true_test.iloc[6288: 6360], 
                       true_test.iloc[6408: 6480], 
                       true_test.iloc[6528: 6600], 
                       true_test.iloc[6648: 6720], 
                       true_test.iloc[6768: 6840], 
                       true_test.iloc[6888: 6960], 
                       true_test.iloc[7008: 7080], 
                       true_test.iloc[7128: 7200], 
                       true_test.iloc[7248: 7320], 
                       true_test.iloc[7368: 7440], 
                       true_test.iloc[7488: 7560], 
                    true_test.iloc[7608: 7680], 
                       ],axis=0)

# 합쳐진 데이터프레임 출력
# true_test = gongjoo_test_AWS_csv[gongjoo_test_AWS_csv['기온(°C)'].isnull()].drop('기온(°C)',axis=1)
# true_test = gongjoo_test_input_csv[gongjoo_test_input_csv['PM2.5'].isnull()].drop('PM2.5',axis=1)
# print('true_test : ',true_test.head(80))
print('true_test.shape:',true_test.shape) # 1 : (7728, 23), 0 : (23184, 9)
print('x_test:',x_test.shape) # 1 : (7728, 23), 0 : (23184, 9)

# 5. 제출 
submission_csv = pd.read_csv(path +'sub_gongjoo.csv')
# print(submission_csv.shape) # (78336, 4)

y_submit = model.predict(true_test)
print('y_submit:',y_submit.shape) # (7728)
print(submission_csv['PM2.5'].shape) # (4608,)
submission_csv['PM2.5'] = y_submit
submission_csv.to_csv(path_save + '0508_01.csv',encoding='utf-8')
