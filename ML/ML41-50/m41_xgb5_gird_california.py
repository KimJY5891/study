import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from pprint import pprint
# 1.데이터 
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=337,train_size=0.8,
)

x_train= RobustScaler().fit_transform(x_train)
x_test= RobustScaler().fit_transform(x_test)
n_split = 5
kfold = KFold(n_splits=n_split)
parameter =  {
    "n_estimators" : [300], # 디폴트 100 / 1 ~ inf / 정수
    "learning_rate" : [0.1], # 디폴트 0.3 / 0 ~ 1 / eta
    "max_depth" : [7], # 디폴트 6 / 0 ~ inf / 정수
    "gamma" : [0], # 디폴트 0 / 0 ~ inf 
    "min_child_weight" : [100], # 디폴트 1 / 0 ~ inf 
    "subsample" : [1], # 디폴트 1 / 0 ~ 1 
    "colsample_bytree" : [0.7], # 디폴트 / 0 ~ 1 
    "colsample_bylevel":[1], # 디폴트 / 0 ~ 1 
    "colsample_bynode":[1], # 디폴트 / 0 ~ 1 
    "reg_alpha":[2], # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
    "reg_lambda":[0.01], # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
}

#2. 모델 
xgb = XGBRegressor(random_state=1234)
model = XGBRegressor(random_state=1234)
# model = GridSearchCV(xgb,parameter,cv=kfold,n_jobs=-1)

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측 
# print("최상의 매개변수 : ",model.best_params_)
# print("최상의 매개변수 : ",model.best_score_)
result = model.score(x_test,y_test)
print("최종점수 : ", result)

'''
그냥 돌렸을 때 
최종점수 :  0.8247401323293999

파라미터 
최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 100, 'n_estimators': 300, 'reg_alpha': 2, 'reg_lambda': 0.01, 'subsample': 1}
최상의 매개변수 :  0.8471597591724406
최종점수 :  0.8353101074886813

파라미터 : 승 
'''
