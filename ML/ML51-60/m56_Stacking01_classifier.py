
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,fetch_california_housing, load_iris, load_wine,load_digits,fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor, XGBClassifier 
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import VotingClassifier,StackingClassifier

dataset_name = [
    '아이리스','캔서','데이콘 디아벳','와인','디짓트','패치콥프타입'
]

for i,v in enumerate(dataset_name) : 
    # 1. 데이터
    if i == 0 :
        x,y = load_breast_cancer(return_X_y=True)
    elif i == 1 :
         x,y = load_breast_cancer(return_X_y=True)
    elif i == 2 :
        path = "./_data/dacon_diabets/"
        path_save = "./_save/dacon_diabets/"
        train_csv=pd.read_csv(path + 'train.csv',index_col=0)
        test_csv=pd.read_csv(path + 'test.csv',index_col=0)
        ####################################### 결측치 처리 #######################################   
        train_csv = train_csv.dropna()
        ####################################### 결측치 처리 #######################################                   
        x = train_csv.drop(['Outcome'],axis=1)
        y = train_csv['Outcome']
    elif i == 3 : 
        x,y = load_wine(return_X_y=True)
    elif i == 4 :
        x,y = load_digits(return_X_y=True)
    else : 
        x,y = fetch_covtype(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,random_state=337,train_size=0.8,shuffle=True, 
        stratify=y
    )
    # print(v,':',x_train.shape)
    scaler = StandardScaler()
    x_train =scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test) 
    #2. 모델 
    xg = XGBClassifier()
    lg = LGBMClassifier()
    knn = KNeighborsClassifier()
    cat = CatBoostClassifier(verbose=0)
    
    model = StackingClassifier(
        estimators=[('XG',xg),('KNN',knn),('CAT',cat)]
    )
    
    model_set = [model,xg,lg,cat,knn]

    for model_i in model_set :  
        # 3. 훈련
        model.fit(x_train,y_train)
        
        # 4. 평가, 예측
        y_pred = model.predict(x_test)
        print('model.score : ',model.score(x_test,y_test))
        print('acc : ',accuracy_score(y_test,y_pred))
        
    
