
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
from sklearn.covariance import EllipticEnvelope

import time
import warnings
warnings.filterwarnings('ignore')
def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    ll = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(ll)
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
col_name = ['Month', 'Day_of_Month', 'Estimated_Departure_Time',
            'Estimated_Arrival_Time', 'Cancelled', 'Diverted', 'Origin_Airport',
            'Origin_Airport_ID', 'Origin_State', 'Destination_Airport',
            'Destination_Airport_ID', 'Destination_State', 'Distance', 'Airline',
            'Cx_trainier_Code(IATA)', 'Cx_trainier_ID(DOT)', 'Tail_Number', 'Delay']
le_col_name = ['Origin_Airport','Origin_State','Destination_Airport','Destination_State','Destination_State'\
        'Airline','Cx_trainier_Code(IATA)','Tail_Number','Delay']
path = "./_data/dacon_airplan/"
path_save = "./_save/dacon_airplan/"

#1. 데이터

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(1000000, 18)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) # y가 없다. train_csv['Delayed','Not_Delayed']
print(test_csv.shape) # (1000000, 17)
print(train_csv.columns)
print(train_csv.info())
print(train_csv.describe())
print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>
####################################### 라벨 인코딩 #####################################
from sklearn.preprocessing import LabelEncoder

#le_data_name = [aaa,bbb,ccc,ddd,eee.]
le = LabelEncoder() # 정의

for i,v in enumerate(le_col_name) :
    #print(np.unique(aaa,return_count=True))
    train_csv[v] = le.fit_transform(train_csv[v]) # 0과 1로 변화 
    # test_csv['Origin_Airport'] = le.transform(test_csv['Origin_Airport'])
    print(v,':',np.unique(train_csv[v],axis=0))

# 결측치 처리
print(train_csv.isna().sum())


# x, y 나누기
x = train_csv.drop(['Delay'])
y = train_csv['Delay']
x_train,x_test, y_train,y_test = train_test_split(
    x,y,
    train_size=0.8,random_state=337,
    stratify=y,shuffle=True,  
)


# 이상치 처리
outliers = EllipticEnvelope(contamination=.3)
outliers.fit(x_train)
x_train = outliers.predict(x_train)
x_test = outliers.predict(x_test)
print(x_train)

