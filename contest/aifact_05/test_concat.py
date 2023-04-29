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
data_csv_list = [
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
from pprint import pprint

#1. 데이터
train_csv = df_concat = pd.concat([gongjo_train_csv, noen_train_csv], axis=0) 
#pprint(train_csv)

train_csv = df_concat = pd.concat([gongjo_train_csv, noen_train_csv], axis=0) 
# train_csv = df_concat.reset_index(drop=True) # 인덱스를 순서대로 자리를 만들어주는 것이 아님 / 인덱스의 번호를 재설정해느 ㄴ것 
# train_csv = df_concat.reindex() # 인덱스를 순서대로 자리를 만들어주는 것이 아님 / 인덱스의 번호를 재설정해는것 
train_csv = train_csv.sort_values(['연도', '일시'])

train_csv.to_csv(path_save+'data_test_sort_values.csv',encoding='cp949')
