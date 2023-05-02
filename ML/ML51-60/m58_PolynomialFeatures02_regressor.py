$ 
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,fetch_california_housing, load_iris, load_wine,load_digits,fetch_covtype,load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

path_dacon_ddarung = "./_data/dacon_ddarung/"
path_kaggle_bike = "./_data/kaggle_bike/"
datasets_names = [
    '디아벳','캘리포니아','따릉이','캐글 바이크'
]
for i, v in enumerate(datasets_names) :
     # 1. 데이터
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
    pf = PolynomialFeatures(degree=2)
    x = pf.fit_transform(x)
    print(x)
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,random_state=337,train_size=0.8,shuffle=True,
        # stratify=y
    )
    scaler = StandardScaler()
    x_train =scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. 모델 

    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    # model= DecisionTreeClassifier()
    model = RandomForestRegressor()

    # 3. 훈련 
    model.fit(x_train,y_train)
    
    # 4. 평가,예측
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    print(v,"의 acc : ",acc)
