
# 수 정 더 필요 
# 회귀 데이터 삭모아서 모델 만들어서 테스트 

import numpy as np
from sklearn.datasets import  load_boston,fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

datasets_name = ['load_boston', 'fetch_california_housing']
model_name= [ 'DecisionTreeRegressor','LogisticRegression','RandomForestRegressor','LinearSVC']
for i in range(4):
    # 1. 데이터 
    if i == 0 or i == 1 : 
        x, y = eval(datasets_name[i])(return_X_y=True)
        print(x.shape, y.shape)
    else : 
        if i == 2 : 
            path_ddarung = './_data/dacon_ddraung/'
            train_csv=pd.read_csv(path_ddarung + 'train.csv',index_col=0)
            train_csv = train_csv.dropna()
            x = train_csv.drop(['count'],axis=1)
            x = np.array(x)
            y = train_csv['count']
        else :    
            path_bike = './_data/kaggle_bike/'
            train_csv = pd.read_csv(path_bike + 'train.csv', index_col=0)
            train_csv = train_csv.dropna()
            x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
            x = np.array(x)
            y = train_csv['count']
            
    for j in range(4):        
        # 2. 모델 
        model = eval(model_name[j])()
        # 3. 컴파일, 훈련 
        model.fit(x, y) # 핏에 컴파일 포함
        # 4. 평가 예측
        results = model.score(x, y)
        print(model_name[j], ':', results)
    
# 분류 acc
# 회귀 r2_스코어
