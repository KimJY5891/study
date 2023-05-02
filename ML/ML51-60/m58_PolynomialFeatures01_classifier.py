import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,fetch_california_housing, load_iris, load_wine,load_digits,fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

dataset_name = [
    '아이리스', '캔서', '데이콘 디아벳', '와인','디짓트','fetch_covtype'
]
for i, v in enumerate(dataset_name) :
     # 1. 데이터
    if i == 0 :
        x,y = load_iris(return_X_y=True)
    elif i== 1 : 
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
        
    pf = PolynomialFeatures(degree=2)
    x = pf.fit_transform(x)
    print(x)
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,random_state=337,train_size=0.8,shuffle=True,
        stratify=y
    )
    scaler = StandardScaler()
    x_train =scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. 모델 

    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    # model= DecisionTreeClassifier()
    model = RandomForestClassifier()

    # 3. 훈련 
    model.fit(x_train,y_train)
    # 4. 평가,예측
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    print(v,"의 acc : ",acc)
