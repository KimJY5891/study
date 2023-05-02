import numpy as np 
import pandas as pd
from sklearn.datasets import load_diabetes,fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

path_dacon_ddarung = "./_data/dacon_ddarung/"
path_kaggle_bike = "./_data/kaggle_bike/"
datasets_names = [
    '디아벳','캘리포니아','따릉이','캐글 바이크'
]
for i, v in enumerate(datasets_names) : 
    #1. 데이터
    if i == 0 :
        x,y =load_diabetes(return_X_y=True)
    elif i == 1 :
        x,y = fetch_california_housing(return_X_y=True)
    elif i == 2 :
        train_csv=pd.read_csv(path_dacon_ddarung + 'train.csv',index_col=0)
        test_csv=pd.read_csv(path_dacon_ddarung + 'test.csv',index_col=0)
        ####################################### 결측치 처리 #######################################
        train_csv = train_csv.dropna()
        ####################################### 결측치 처리 #######################################
        x = train_csv.drop(['count'],axis=1)#
        y = train_csv['count']
    else : 
        train_csv=pd.read_csv(path_kaggle_bike + 'train.csv',index_col=0)
        test_csv=pd.read_csv(path_kaggle_bike + 'test.csv',index_col=0)
        ####################################### 결측치 처리 ####################################### 
        train_csv = train_csv.dropna()
        ####################################### 결측치 처리 ####################################### 
        x = train_csv.drop(['count'],axis=1)#
        y = train_csv['count']
        
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,random_state=337,train_size=0.8,shuffle=True, 
        # stratify=y
    )
    
    scaler = StandardScaler()
    x_train =scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. 모델 
    xg = XGBRegressor()
    lg = LGBMRegressor()
    knn = KNeighborsRegressor(n_neighbors=8)
    cb = CatBoostRegressor(verbose=0)
    
    model = VotingRegressor(
        estimators=[('XG',xg),('LG',lg),('KNN',knn)], 

    )
    model_set = [model,xg,lg,cb]
    for model_i in model_set :
        # 3. 훈련 
        model_i.fit(x_train,y_train)
        # 4. 평가
        y_pred = model_i.predict(x_test)
        score2 = r2_score(y_test,y_pred)
        class_name=model_i.__class__.__name__
        print(v,"{0} r2 : {1:4f}".format(class_name,score2))
'''
디아벳 VotingRegressor r2 : 0.388146
디아벳 XGBRegressor r2 : 0.161307
디아벳 LGBMRegressor r2 : 0.312648
디아벳 CatBoostRegressor r2 : 0.398961
=> 디아벳 VotingRegressor r2 : 0.388146
캘리포니아 VotingRegressor r2 : 0.822444
캘리포니아 XGBRegressor r2 : 0.827001
캘리포니아 LGBMRegressor r2 : 0.832710
캘리포니아 CatBoostRegressor r2 : 0.843305
=>캘리포니아 CatBoostRegressor r2 : 0.843305
따릉이 VotingRegressor r2 : 0.804587
따릉이 XGBRegressor r2 : 0.813301
따릉이 LGBMRegressor r2 : 0.798627
따릉이 CatBoostRegressor r2 : 0.823863
=>따릉이 CatBoostRegressor r2 : 0.823863
캐글 바이크 VotingRegressor r2 : 0.997331
캐글 바이크 XGBRegressor r2 : 0.999583
캐글 바이크 LGBMRegressor r2 : 0.999509
캐글 바이크 CatBoostRegressor r2 : 0.999647
=>캐글 바이크 CatBoostRegressor r2 : 0.999647
'''
