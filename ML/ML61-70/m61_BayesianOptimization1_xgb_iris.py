import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor, XGBClassifier 


# 1. 데이터 

x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337,train_size=0.8,shuffle=True,
    stratify=y
)

scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 

parameter =  {
    "n_estimators" : (100,600), # 디폴트 100 / 1 ~ inf / 정수
    "learning_rate" : (0.1,0.3), # 디폴트 0.3 / 0 ~ 1 / eta
    "max_depth" : (0,10), # 디폴트 6 / 0 ~ inf / 정수
    "gamma" :(0,10), # 디폴트 0 / 0 ~ inf 
    "min_child_weight" : (0,100), # 디폴트 1 / 0 ~ inf 
    "subsample" : (0,1), # 디폴트 1 / 0 ~ 1 
    "colsample_bytree" : (0,1), # 디폴트 / 0 ~ 1 
    "colsample_bylevel": (0,1), # 디폴트 / 0 ~ 1 
    "colsample_bynode":(0,1), # 디폴트 / 0 ~ 1 
    "reg_alpha":(0,10), # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
    "reg_lambda":(0,10), # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
    "random_state" : (0,10000)
}
def model_function(n_estimators,learning_rate,max_depth,gamma,
                   min_child_weight,subsample,colsample_bytree,colsample_bylevel,
                   colsample_bynode,reg_alpha,reg_lambda,random_state) : 
    parameter =  {
        "n_estimators" : int(round(n_estimators)), # 디폴트 100 / 1 ~ inf / 정수
        "learning_rate" : learning_rate, # 디폴트 0.3 / 0 ~ 1 / eta
        "max_depth" : int(round(max_depth)), # 디폴트 6 / 0 ~ inf / 정수
        "gamma" :gamma, # 디폴트 0 / 0 ~ inf 
        "min_child_weight" : min_child_weight, # 디폴트 1 / 0 ~ inf 
        "subsample" : subsample, # 디폴트 1 / 0 ~ 1 
        "colsample_bytree" : colsample_bytree, # 디폴트 / 0 ~ 1 
        "colsample_bylevel": colsample_bylevel, # 디폴트 / 0 ~ 1 
        "colsample_bynode":colsample_bynode, # 디폴트 / 0 ~ 1 
        "reg_alpha":reg_alpha, # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
        "reg_lambda":reg_lambda, # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
        "random_state" : random_state
    }
    model = XGBClassifier(**parameter)
    # 3. 훈련
    model.fit(
        x_train,y_train,
        eval_set=[(x_train,y_train),(x_test,y_test)],
        verbose=0
              )
    # 4. 평가 예측  
    result = model.score(x_test,y_test)
    print('최종점수 : ',result)
    y_pred = model.predict(x_test)
    print('acc : ', accuracy_score(y_test,y_pred))

from bayes_opt import BayesianOptimization
import time
start= time.time()
lgb_bo = BayesianOptimization(
    f=model_function,
    pbounds= parameter,
    random_state=337
)
lgb_bo.maximize(
    init_points=5,
    n_iter=100
)
# 최소값은 앞에 -1을 곱하면 된다. 
# 

end= time.time()
print('시간 : ',round(end-start,3))
