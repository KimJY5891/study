import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. 데이터
x,y = load_breast_cancer(return_X_y=True)

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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt =DecisionTreeClassifier()

## 삼대장 ##
from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostRegressor,CatBoostClassifier

xg = XGBClassifier()
lg = LGBMClassifier()
cb = CatBoostClassifier(verbose=0)
    
#  이 코딩에서는 
# model= DecisionTreeClassifier()
# model = VotingClassifier(
li = []
models = [xg,lg,cb]
for model in models :
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_pred.reshape(y_pred.shape[0],1)
    li.append(y_pred)
    score = accuracy_score(y_test,y_pred)
    class_name =model.__class__.__name__
    print("{0} acc : {1:.4f}".format(class_name,score))
# print(li)
y_stacking_pred = np.concatenate(li,axis=1) # 행렬(2차원) 형태로 concat 해줘야하는데, 벡터형태로 받아서 안맞는것 
print(y_stacking_pred.shape)
model = CatBoostClassifier(verbose=0)
model.fit(y_stacking_pred,y_test)
score = model.score(y_stacking_pred,y_test)
print('스태킹 결과 : ', score)
