import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score,r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings 

# 1. 데이터
path = "./_data/dacon_ddarung/"
path_save = "./_save/dacon_ddarung/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0) # class : padas
print(train_csv.columns) 
'''
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',        
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object')
'''
####################################### 결측치 처리 #######################################                                                                                 #
imputer = IterativeImputer(estimator=XGBRegressor()) 
train_csv=imputer.fit_transform(train_csv) # <class 'numpy.ndarray'>
print(train_csv.shape) # (1459, 10)
####################################### 결측치 처리 #######################################     
# 판다스로 변환         
train_csv = pd.DataFrame(train_csv, columns=['hour', 'hour_bef_temperature', 'hour_bef_precipitation',        
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'])
# x, y 나누기 
x = train_csv.drop(['count'],axis=1)#
y = train_csv['count'] 
x_train, x_test, y_train, y_test = train_test_split(
            x,y, train_size=0.8, shuffle=True, random_state=337
        )
# 스케일러
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits = n_splits,
      shuffle=True,  
      random_state=123,
      )
paramiters = [
    {"n_estimators" : [100,200,300],"learning_rate" : [0.1,0.3,0.001,0.01],"max_depth" : [4,5,6]},
    {"n_estimators" : [90,100,110],"learning_rate" : [0.1,0.001,0.01],"max_depth" : [4,5,6],"colsample_bytree" : [0.6,0.9,1]},
    {"n_estimators" : [90,110],"learning_rate" : [0.1,0.001,0.5],"max_depth" : [4,5,6],"colsample_bytree" : [0.6,0.9,1],"colsample_bylevel":[0.6,0.7,0.9]},
]

# 2. 모델

model = GridSearchCV(XGBRegressor(
                        n_jobs = -1,                        
                        tree_method='gpu_hist',
                        predictor='gpu_predictor',
                        gpu_id=0
                    ),
                    paramiters,
                    cv=5,
                    verbose=1
                )

# 3. 컴파일, 훈련
model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_) 
print("최적의 파라미터 : ",model.best_params_)
print("best_score_ : ",model.best_score_) # train의 베스트 스코어 
print("model.score : ",model.score(x_test,y_test)) # test의 베스트 스코어 

# 4. ,평가, 예측
result = model.score(x_test,y_test)
print('result:',result)
y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print('r2_score : ',r2 )
'''
랜덤
최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1}
best_score_ :  0.782112921158777
model.score :  0.7751968694554608
result: 0.7751968694554608
r2_score :  0.7751968694554608
'''
'''
그리드
최적의 파라미터 :  {'colsample_bylevel': 0.7, 'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 110}
best_score_ :  0.791126357871455
model.score :  0.783282472909788
result: 0.783282472909788
r2_score :  0.783282472909788
'''
