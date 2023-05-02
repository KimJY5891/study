# 
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

dataset_name = [
    '아이리스', '캔서', '데이콘 디아벳', '와인','디짓트','fetch_covtype'
]

for i, v in enumerate(dataset_name) :
    # 1. 데이터
    if i ==0  : 
        x,y = load_iris(return_X_y=True)
    elif i==1 : 
        x,y = load_breast_cancer(return_X_y=True)
    elif i ==2 : 
        path = "./_data/dacon_diabets/"
        path_save = "./_save/dacon_diabets/"
        train_csv=pd.read_csv(path + 'train.csv',index_col=0)
        test_csv=pd.read_csv(path + 'test.csv',index_col=0)
        ####################################### 결측치 처리 #######################################   
        train_csv = train_csv.dropna()
        ####################################### 결측치 처리 #######################################                   
        x = train_csv.drop(['Outcome'],axis=1)
        y = train_csv['Outcome']
    elif i ==3 : 
        x,y = load_wine(return_X_y=True)
    elif i ==4 :
        x,y = load_digits(return_X_y=True)
    else : 
        x,y = fetch_covtype(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,random_state=337,train_size=0.8,shuffle=True, 
        stratify=y
    )
    scaler = StandardScaler()
    x_train =scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. 모델 

    xg = XGBClassifier()
    lg = LGBMClassifier()
    knn = KNeighborsClassifier(n_neighbors=8)
    cb = CatBoostClassifier(verbose=0)

    # model= DecisionTreeClassifier()
    model = VotingClassifier(
        # estimators = [('XG',xg),('LG',lg),('CB',cb)],
        estimators=[('XG',xg),('LG',lg),('KNN',knn)], 
            voting = 'hard', 
            # voting = 'soft', # regressor에서 voting 하이퍼 파라미터가 안먹힌다. 
    )
    model_set = [model,xg,lg,cb,knn]
    for model_i in model_set :
        model_i.fit(x_train,y_train)
        y_pred = model_i.predict(x_test)
        score2= accuracy_score(y_test,y_pred)
        class_name = model_i.__class__.__name__
        print(v,"{0} acc : {1:.4f}".format(class_name,score2))
'''

아이리스 VotingClassifier acc : 0.9333
아이리스 XGBClassifier acc : 0.9333
아이리스 LGBMClassifier acc : 0.9333
아이리스 CatBoostClassifier acc : 0.9333  
아이리스 KNeighborsClassifier acc : 0.9667
=> KNeighborsClassifier acc : 0.9667

캔서 VotingClassifier acc : 0.9649
캔서 XGBClassifier acc : 0.9474
캔서 LGBMClassifier acc : 0.9561
캔서 CatBoostClassifier acc : 0.9649
캔서 KNeighborsClassifier acc : 0.9737
=>KNeighborsClassifier acc : 0.9737

데이콘 디아벳 VotingClassifier acc : 0.7634
데이콘 디아벳 XGBClassifier acc : 0.7634
데이콘 디아벳 LGBMClassifier acc : 0.7328
데이콘 디아벳 CatBoostClassifier acc : 0.7786
데이콘 디아벳 KNeighborsClassifier acc : 0.7405
=>CatBoostClassifier acc : 0.7786

와인 VotingClassifier acc : 1.0000
와인 XGBClassifier acc : 1.0000
와인 LGBMClassifier acc : 1.0000
와인 CatBoostClassifier acc : 1.0000
와인 KNeighborsClassifier acc : 0.9444
=> 와인 VotingClassifier acc : 1.0000
XGBClassifier acc : 1.0000
LGBMClassifier acc : 1.0000
CatBoostClassifier acc : 1.0000


디짓트 VotingClassifier acc : 0.9778
디짓트 XGBClassifier acc : 0.9556
디짓트 LGBMClassifier acc : 0.9778
디짓트 CatBoostClassifier acc : 0.9861
디짓트 KNeighborsClassifier acc : 0.9750
=> CatBoostClassifier acc : 0.9861

결론 의외로 보팅이 좋을 수도 있고 아닐 수도 있다.

'''
