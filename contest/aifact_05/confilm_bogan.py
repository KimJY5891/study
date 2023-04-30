# 라벨 인코딩# 시계열
import numpy as np
import pandas as pd
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
import pprint
import time
import warnings
warnings.filterwarnings('ignore')
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
col_name = ['연도','일시', '측정소', 'PM2.5']
le_col_name = ['일시', '측정소']
path = "./_data/aifact_05/"
path_train = "./_data/aifact_05/TRAIN/"
path_save = "./_save/aifact_05/"
path_data_imputer = "./_data/aifact_05/TRAIN/"
data_name_list = [
    '공주.csv','노은동.csv','논산.csv','대천2동.csv','독곶리.csv','동문동.csv',
    '모종동.csv','문창동.csv','성성동.csv','신방동.csv','신흥동.csv','아름동.csv','예산군.csv',
    '읍내동.csv','이원면.csv','정림동.csv','홍성읍.csv']
gongjo_train_csv=pd.read_csv(path_train +'공주.csv',encoding='cp949')
noen_train_csv=pd.read_csv(path_train +'노은동.csv',encoding='cp949')
nonsan_train_csv=pd.read_csv(path_train +'논산.csv',encoding='cp949')
deacheon_train_csv=pd.read_csv(path_train +'대천2동.csv',encoding='cp949')
doggojlee_train_csv=pd.read_csv(path_train +'독곶리.csv',encoding='cp949')
dongmoon_train_csv=pd.read_csv(path_train +'동문동.csv',encoding='cp949')
mojong_train_csv=pd.read_csv(path_train +'모종동.csv',encoding='cp949')
moonchang_train_csv=pd.read_csv(path_train +'문창동.csv',encoding='cp949')
sungsung_train_csv=pd.read_csv(path_train +'성성동.csv',encoding='cp949')
shinbang_train_csv=pd.read_csv(path_train +'신방동.csv',encoding='cp949')
shinghng_train_csv=pd.read_csv(path_train +'신흥동.csv',encoding='cp949')
aruen_train_csv=pd.read_csv(path_train +'아름동.csv',encoding='cp949')
yeshan_train_csv=pd.read_csv(path_train +'예산군.csv',encoding='cp949')
epnea_train_csv=pd.read_csv(path_train +'읍내동.csv',encoding='cp949')
twoone_train_csv=pd.read_csv(path_train +'이원면.csv',encoding='cp949')
jungleem_train_csv=pd.read_csv(path_train +'정림동.csv',encoding='cp949')
hongsung_train_csv=pd.read_csv(path_train +'홍성읍.csv',encoding='cp949')
data_csv_list = [
    gongjo_train_csv,noen_train_csv,nonsan_train_csv,deacheon_train_csv,doggojlee_train_csv,
    dongmoon_train_csv,mojong_train_csv,moonchang_train_csv,sungsung_train_csv,shinbang_train_csv,
    shinghng_train_csv,aruen_train_csv,yeshan_train_csv,epnea_train_csv,twoone_train_csv,
    jungleem_train_csv,hongsung_train_csv
]

'''
할 것
1. 데이터 전부 결측치 처리
2. 스케일링 하기 
'''
# 1. 데이터

print(gongjo_train_csv.columns)
# Index(['일시', '측정소', 'PM2.5'], dtype='object')
# 데이터 타입 확인 
for i in col_name :
    df = gongjo_train_csv[i]
    print(f' gongjo_train_csv_{i}',df.dtypes)
# 판다스로 데이터 타입 확인하는 방법  -> .dtypes
'''
# 데이터 타입 확인 
Index(['일시', '측정소', 'PM2.5'], dtype='object')
gongjo_train_csv_일시 object
gongjo_train_csv_측정소 object
gongjo_train_csv_PM2.5 float64
gongjo_train_csv[일시] <class 'pandas.core.series.Series'>
gongjo_train_csv[측정소] <class 'pandas.core.series.Series'>
gongjo_train_csv[PM2.5] <class 'pandas.core.series.Series'>
''' 
# 라벨 인코딩
# 1) 측정소만 하기
le = LabelEncoder() 

for i,v in enumerate(le_col_name) :
    for i2, v2 in enumerate(data_csv_list) :
        v2[v] = le.fit_transform(gongjo_train_csv[v])
        print(f'{v2}의{v} :', np.unique(v2[v],axis=0))     
'''
# 데이터 타임 가능할 경우 사용할 코드
gongjo_train_csv['측정소'] = le.fit_transform(gongjo_train_csv['측정소'])
print('측정소 : ',np.unique(gongjo_train_csv['측정소'],axis=0)) # 0 = 공주
    gongjo_train_csv = pd.DataFrame(gongjo_train_csv, columns=col_name)
    gongjo_train_csv['측정소'] = le.fit_transform(gongjo_train_csv['측정소'])
    print('측정소 : ',np.unique(gongjo_train_csv['측정소'],axis=0))
# 2) 숫자형으로 인식 가능한 날짜데이터로 변경하기
# 데이터 타임 바꾸기 
# for i,v in enumerate() : 
start_time = time.time() 
dates = gongjo_train_csv['일시']
# print(gongjo_train_csv['일시'])
# pd.to_datetime('2022-04-27', format='%Y-%m-%d') # 낱개의 방법  
gongjo_train_csv['일시'] 
dt= pd.to_datetime("01-15 11:00", format="%m-%d %H:%M") 
# gongjo_train_csv = pd.to_datetime(gongjo_train_csv['일시'], format='%Y-%m-%d') # csv 데이터일 경우 방법
print('dt : ',dt) # csv 데이터일 경우 방법
# dates =pd.to_datetime(dates)
# print(dates)
# dates['일시'] = pd.to_datetime(dates['일시'], format='%Y')
# gongjo_train_csv['일시'] = dates
# print(gongjo_train_csv['일시'])
end_time = time.time() 
print('시간 : ', round(end_time - start_time,2))
'''
#결측치 처리 -> 나중에 ffill이랑 dfill으로 하기
imputer = IterativeImputer(estimator=XGBRegressor())  

# for i,v in enumerate(data_csv_list) : 
#     v = imputer.fit_transform(v).
# for i,v in enumerate(data_csv_list) : 
#     v['PM2.5'] = v.interpolate(v)
# 모두 한꺼번에 하기 
for i,v in enumerate(col_name) :
    for i2, v2 in enumerate(data_csv_list) : 
        v2[v] = v2[v].interpolate(order=3)
        v2.to_csv(path_data_imputer+f'{v2}_le_pd_inter.csv',encoding='UTF-8',index=False) 
        
# gongjo_train_csv['PM2.5'] = gongjo_train_csv['PM2.5'].interpolate(order=3)

    
'''
# 한 개만 가능한거 
imputer = IterativeImputer(estimator=XGBRegressor())  
gongjo_train_csv = imputer.fit_transform(gongjo_train_csv)
gongjo_train_csv = pd.DataFrame(gongjo_train_csv, columns=col_name)
gongjo_train_csv['측정소'] = le.fit_transform(gongjo_train_csv['측정소'])
print('측정소 : ',np.unique(gongjo_train_csv['측정소'],axis=0)) # 0 = 공주
'''
 
# 시계열 데이터  모델 짜는 방법 다시 보기 
# 3) x와 y나누기  
# 4) 스케일링 하기 



# 저장 
# gongjo_train_csv.to_csv(path_data_imputer+'공주_nonele02.csv',encoding='UTF-8',index=False) 

