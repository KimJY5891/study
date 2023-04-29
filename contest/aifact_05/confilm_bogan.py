# 시계열
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
import time
import warnings
warnings.filterwarnings('ignore')
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 

path = "./_data/aifact_05/"
path_train = "./_data/aifact_05/TRAIN/"
path_save = "./_save/aifact_05/"

data_name_list = [
    '공주.csv','노은동.csv','논산.csv','대천2동.csv','독곶리.csv','동문동.csv',
    '모종동.csv','문창동.csv','성성동.csv','신방동.csv','신흥동.csv','아름동.csv','예산군.csv',
    '읍내동.csv','이원면.csv','정림동.csv','홍성읍.csv']
gongjo_train_csv=pd.read_csv(path_train +'공주.csv',index_col=0)
noen_train_csv=pd.read_csv(path_train +'노은동.csv',index_col=0)
nonsan_train_csv=pd.read_csv(path_train +'논산.csv',index_col=0)
deacheon_train_csv=pd.read_csv(path_train +'대천2동.csv',index_col=0)
doggojlee_train_csv=pd.read_csv(path_train +'독곶리.csv',index_col=0)
dongmoon_train_csv=pd.read_csv(path_train +'동문동.csv',index_col=0)
mojong_train_csv=pd.read_csv(path_train +'모종동.csv',index_col=0)
moonchang_train_csv=pd.read_csv(path_train +'문창동.csv',index_col=0)
sungsung_train_csv=pd.read_csv(path_train +'성성동.csv',index_col=0)
shinbang_train_csv=pd.read_csv(path_train +'신방동.csv',index_col=0)
shinghng_train_csv=pd.read_csv(path_train +'신흥동.csv',index_col=0)
aruen_train_csv=pd.read_csv(path_train +'아름동.csv',index_col=0)
yeshan_train_csv=pd.read_csv(path_train +'예산군.csv',index_col=0)
epnea_train_csv=pd.read_csv(path_train +'읍내동.csv',index_col=0)
twoone_train_csv=pd.read_csv(path_train +'이원면.csv',index_col=0)
jungleem_train_csv=pd.read_csv(path_train +'정림동.csv',index_col=0)
hongsung_train_csv=pd.read_csv(path_train +'홍성읍.csv',index_col=0)
data_csv_list = [
    gongjo_train_csv,noen_train_csv,nonsan_train_csv,deacheon_train_csv,doggojlee_train_csv,
    dongmoon_train_csv,mojong_train_csv,moonchang_train_csv,sungsung_train_csv,shinbang_train_csv,
    shinghng_train_csv,aruen_train_csv,yeshan_train_csv,epnea_train_csv,twoone_train_csv,
    jungleem_train_csv,hongsung_train_csv
]

# 0. 결측치, 이상치 처리
train_csv = pd.read_csv(path_save + 'train.csv',index_col= 0, encoding='cp949')
for i,v in enumerate(data_csv_list) : 
    print(data_name_list[i],'의 결측치는 ', v.isnull().sum())

'''
공주의 결측치는
일시         0
측정소        0
PM2.5    770
dtype: int64

노은동 의 결측치는
일시         0
측정소        0
PM2.5    924
dtype: int64

논산의 결측치는
일시          0
측정소         0
PM2.5    1187
dtype: int64

대천2동의 결측치는
일시         0
측정소        0
PM2.5    706
dtype: int64

독곶리의 결측치는
일시          0
측정소         0
PM2.5    1031
dtype: int64

동문동의 결측치는
일시          0
측정소         0
PM2.5    2455
dtype: int64

모종동의 결측치는
일시          0
측정소         0
PM2.5    1049
dtype: int64

문창동의 결측치는
일시         0
측정소        0
PM2.5    636
dtype: int64

성성동의 결측치는
일시         0
측정소        0
PM2.5    805
dtype: int64

신방동의 결측치는
일시         0
측정소        0
PM2.5    172
dtype: int64

신흥동의 결측치는
일시         0
측정소        0
PM2.5    759
dtype: int64

아름동의 결측치는
일시         0
측정소        0
PM2.5    798
dtype: int64

예산군의 결측치는
일시         0
측정소        0
PM2.5    764
dtype: int64

읍내동의 결측치는
일시         0
측정소        0
PM2.5    882
dtype: int64

이원면의 결측치는
일시          0
측정소         0
PM2.5    1195
dtype: int64

정림동의 결측치는
일시         0
측정소        0
PM2.5    680
dtype: int64

홍성읍의 결측치는
일시         0
측정소        0
PM2.5    729
dtype: int64

'''
