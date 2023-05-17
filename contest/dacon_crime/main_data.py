# 피처 임포 - > 코릴레이션 -> 다중공산성이랑 비슷하다.
import math
import numpy as np
import pandas as pd
import glob
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, OneHotEncoder, PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score, f1_score
from catboost import CatBoostClassifier,CatBoostRegressor
from hyperopt import hp, fmin, tpe, Trials
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import time
import warnings
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
data_list = ['월', 
             '요일','시간', '소관경찰서', '소관지역','사건발생거리', 
             '강수량(mm)', '강설량(mm)','적설량(cm)', '풍향','안개', 
            '짙은안개', '번개', '진눈깨비', '서리','연기/연무', 
            '눈날림','범죄발생지', 'TARGET']
outliers_list = [ 
                 '사건발생거리','강수량(mm)', '강설량(mm)','적설량(cm)', 
'짙은안개', '번개', '진눈깨비', '서리','연기/연무',
'눈날림','범죄발생지']
path = "c:/study/_data/dacon_crime/"
path_save = "c:/study/_save/dacon_crime/"
'''
def find_negative_one(lst):
    return [i for i, x in enumerate(lst) if x == -1]   

def find_consecutive(lst):
    result = []
    negative_indices = find_negative_one(lst)  # Find indices where the value is -1
    
    if len(negative_indices) == 0:
        return result  # Return empty result if no -1 values are found
    
    i = 0
    while i < len(negative_indices):
        start = negative_indices[i]
        end = start
        while i+1 < len(negative_indices) and negative_indices[i+1] == end+1:
            end = negative_indices[i+1]
            i += 1
        i += 1
        if end - start > 0:
            result.append((start, end))
    
    return result
'''
def find_negative_one(lst):
    return [i for i, x in enumerate(lst) if x == -1]   




#1. 데이터

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(871393, 9)

test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) # y가 없다. train_csv['Calories_Burned']
print(test_csv.shape) # (159621, 8)

# print(train_csv.info()) 
# print(train_csv.corr())


print(train_csv)
'''
#   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   월        84406 non-null  int64
 1   요일       84406 non-null  object
 2   시간       84406 non-null  int64
 3   소관경찰서    84406 non-null  int64
 4   소관지역     84406 non-null  float64
 5   사건발생거리   84406 non-null  float64
 6   강수량(mm)  84406 non-null  float64
 9   풍향       84406 non-null  float64
 10  안개       84406 non-null  float64
 11  짙은안개     84406 non-null  float64
 12  번개       84406 non-null  float64
 13  진눈깨비     84406 non-null  float64
 14  서리       84406 non-null  float64
 15  연기/연무    84406 non-null  float64
 16  눈날림      84406 non-null  float64
 17  범죄발생지    84406 non-null  object
 18  TARGET   84406 non-null  int64
dtypes: float64(13), int64(4), object(2)
print(train_csv.columns)
Index(['월', '요일', '시간', '소관경찰서', '소관지역', '사건발생거리', '강수량(mm)', '강설량(mm)',
       '적설량(cm)', '풍향', '안개', '짙은안개', '번개', '진눈깨비', '서리', '연기/연무', '눈날림',
       '범죄발생지', 'TARGET'],
      dtype='object')
      
    

''' 
# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의
le_list = ['요일', '범죄발생지']
csv_all = pd.concat([train_csv,test_csv],axis=0)
print(csv_all.columns)
for i,v in enumerate(le_list) :
    csv_all[v] = le.fit_transform(csv_all[v])
    train_csv[v] = le.transform(train_csv[v]) 
    test_csv[v] = le.transform(test_csv[v])
    print(f'{i}번째',np.unique(csv_all[v]))



# 이상치 처리
out_list = ['사건발생거리','강수량(mm)', '강설량(mm)','적설량(cm)',
'짙은안개', '번개', '진눈깨비', '서리','연기/연무',
'눈날림','범죄발생지']
# '사건발생거리','강수량(mm)', '강설량(mm)','적설량(cm)', 5 6 7 8
# '짙은안개', '번개', '진눈깨비', '서리','연기/연무', 11 12 13 14 15
# '눈날림','범죄발생지',16 17

'''
EllipticEnvelope은 이상치 탐지 알고리즘으로, 
탐지된 이상치를 1로 표시하고 정상값을 -1로 표시합니다. 
예측 결과가 1이라면 해당 데이터는 이상치로 판단된 것이고,
-1이라면 정상값으로 판단된 것입니다.
'''
out_list = ['사건발생거리','강수량(mm)', '강설량(mm)','적설량(cm)',
'짙은안개', '번개', '진눈깨비', '서리','연기/연무',
'눈날림','범죄발생지']
'''
outliers_variables = {}  # Dictionary to store the variables
for i,v in enumerate(outliers_list):
    variable_name = f"index_{i}"
    variable_value = i * 2
    outliers_variables[variable_name] = variable_value

# Access the dynamically created variables
for variable_name, variable_value in outliers_variables.items():
    print(f"{variable_name}: {variable_value}")
'''
outliers = EllipticEnvelope(contamination=.2)
for i,v in enumerate(outliers_list) :
    result = []
    train_data = train_csv[v].copy()  
    train_data = train_data.values.reshape(-1,1)  
    train_data_out = outliers.fit_predict(train_data).astype(float)
    result = find_negative_one(train_data_out)
    for rs_i,rs_v in enumerate(result) :
        print(rs_i,'번째')
        train_csv.at[rs_v,v] = np.nan
# for i, v in enumerate(outliers_list):
#     print(i,'번째 실행')
#     train_data = train_csv[v].copy()  
#     train_data = train_data.values.reshape(-1,1)  
#     train_data_out = outliers.fit_predict(train_data).astype(float)
#     print(train_data_out)
#     result = find_negative_one(train_data_out)
#     print(result)

################################## 결측치 처리 #################################
for i,v in enumerate(outliers_list) :
    train_csv[v] = train_csv[v].interpolate(order=2)
    print(train_csv[v].isna().sum())


train_csv = pd.DataFrame(train_csv,columns=data_list)
print('train_csv:',train_csv)
print('train_csv:',train_csv.shape)
# drop
# test_csv = test_csv.drop(['시간','요일','풍향',],axis=1)
# x = train_csv.drop(['TARGET','시간','요일','풍향',],axis=1)

x = train_csv.drop(['TARGET',],axis=1)
pca = PCA(n_components=15)
'''
상관관계가 높은게 2개라면 서로 비율이 안맞아서 과적합 될 수도 있는 가능성을 없애느 ㄴ것이다. 
17
acc : 0.5470915768273901
f1 : 0.5470915768273901
16
[0 0 0 ... 0 2 0]
acc : 0.5460253524463926
f1 : 0.5460253524463926
걸린시간: 69.77
15

'''
x = pca.fit_transform(x)
test_csv = pca.transform(test_csv)
print("x : ", x) 


y = train_csv['TARGET']
print('smote 전',y.shape) #  (84406,)
count00 = 0
count01 = 0
count02 = 0
for label in y:
    if label == 0:
        count00 += 1
    if label == 1:
        count01 += 1
    if label == 2:
        count02 += 1
print('전')
print(count00) # 36453
print(count01) # 25397
print(count02) # 22556

smote = SMOTE(
    k_neighbors = 10,
    random_state = 8715,
) # 분류의 경우 내가 숫자로 지정하는게 아니라 이런식으로 지정한 다음에 
# sampling_strategy = {0: 10000, 1: 20000, 2: 30000} 안에 변수명으로 설정해줘야함 
x, y = smote.fit_resample(x, y)
# y = to_categorical(train_csv['TARGET'])
print('smote 후',y.shape) # (109359,)
count00 = 0
count01 = 0
count02 = 0
for label in y:
    if label == 0:
        count00 += 1
    if label == 1:
        count01 += 1
    if label == 2:
        count02 += 1
print('후')
print(count00) # 36453              | 파라미터 적용 후 : 
print(count01) # 36453 # 5천개 증가 | 파라미터 적용 후 : 
print(count02) # 36453 # 7천개 증가 | 파라미터 적용 후 : 

# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)

# vif = pd.DataFrame()
# vif['variables'] = x.columns
# vif['vif'] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
# print(vif)
'''

   variables       vif
0            월  1.282735
1          요일  1.032687
2          시간  1.000298
3    소관경찰서  1.850206
4      소관지역  1.844475
5  사건발생거리  1.014597
6    강수량(mm)  1.270713
7    강설량(mm)  1.740622
8    적설량(cm)  1.679097
9          풍향  1.056074
10         안개  1.595804
11     짙은안개  1.217778
12         번개  1.330880
13     진눈깨비  1.573900
14         서리  1.640009
15    연기/연무  1.262941
16       눈날림  1.439312
17   범죄발생지  1.001912

'''

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715,stratify=y
)
