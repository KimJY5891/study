    # "n_estimators" : [100,200,300,400,500,600], # 디폴트 100 / 1 ~ inf / 정수
    # "learning_rate" : [0.1,0.2,0.3,0.5,1,0.01,0.001], # 디폴트 0.3 / 0 ~ 1 / eta
    # "max_depth" : [None,2,3,4,5,6,7,8,9,10], # 디폴트 6 / 0 ~ inf / 정수
    # "gamma" : [0,1,2,3,4,5,7,8,9,10], # 디폴트 0 / 0 ~ inf 
    # "min_child_weight" : [0,0.1,0.01,0.001,0.5,1,5,10,100], # 디폴트 1 / 0 ~ inf 
    # "subsample" : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0 ~ 1 
    # "colsample_bytree" : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 / 0 ~ 1 
    # "colsample_bylevel":[0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 / 0 ~ 1 
    # "colsample_bynode":[0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 / 0 ~ 1 
    # "reg_alpha":[0,0.1,0.01,0.001,1,2,10], # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
    # "reg_lambda":[0,0.1,0.01,0.001,1,2,10], # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler

# 1.데이터 
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=337,train_size=0.8,stratify=y
)


x_train= RobustScaler().fit_transform(x_train)
x_test= RobustScaler().fit_transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split)
parameter =  {
    "n_estimators" : 100, # 디폴트 100 / 1 ~ inf / 정수
    "learning_rate" : 0.3, # 디폴트 0.3 / 0 ~ 1 / eta
    "max_depth" : 2, # 디폴트 6 / 0 ~ inf / 정수
    "gamma" :0, # 디폴트 0 / 0 ~ inf 
    "min_child_weight" : 1, # 디폴트 1 / 0 ~ inf 
    "subsample" : 0.3, # 디폴트 1 / 0 ~ 1 
    "colsample_bytree" : 1, # 디폴트 / 0 ~ 1 
    "colsample_bylevel": 1 , # 디폴트 / 0 ~ 1 
    "colsample_bynode":1, # 디폴트 / 0 ~ 1 
    "reg_alpha":0, # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
    "reg_lambda":1, # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
    "random_state" : 337
}
# xgboost.core.XGBoostError: Invalid Parameter format for colsample_bynode expect float but value='[1]'
# 자꾸 이런게, 리스트 형태 = 2개 이상이라서 
# 왜냐면 서치와 다르게 여러개 작성시에는 여러 개의 값을 넣는다.
# 하나씩 넣어서 보여주는게 아니라 
# 그리드 서치 없이 모델로 변수를 넣어서 작성했을 때는 여러개 넣기 때문에 
# 값의 갯수를 하나로 변경해서 넣어줘야한다.
# 정의 되지 않는 파람도 실행은 되지만 적용 되지는 않는다. 

#2. 모델 
model = XGBClassifier(**parameter)

# 3. 훈련
# model.set_params(early_stopping_rounds = 10,**parameter) 도 가능하다. 
model.fit(
    x_train,y_train,
    eval_set =[(x_train,y_train),(x_test,y_test)], # 발리데이션 데이터가 되는 것이다. 
        early_stopping_rounds = 10, # n_estimators의 파라미터가 커짐
    # TypeError: fit() got an unexpected keyword argument 'early_stopping_round
    # 발리데이션 데이터 셋을 넣으라는 의미
    verbose= True # 과정이 다 나옴 
    # verbose= False # 최종점수만 나옴 
    # verbose= 0 # 최종점수만 나옴 
    # verbose= 1 # 과정이 다 나옴 # 3도 가능 하다. 0만 최종점수 나오는 것이다. 
)

# 4. 평가, 예측 
# print("최상의 매개변수 : ",model.best_params_)
# print("최상의 매개변수 : ",model.best_score_)
result = model.score(x_test,y_test)
print("최종점수 : ", result)

'''

파라미터 안 사용 
최종점수 :  0.8947368421052632

파라미터 사용 
최상의 매개변수 :  {'colsample_bylevel': 1, 'colsample_bynode': 
1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.3, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.3}
최상의 매개변수 :  0.9758241758241759
최종점수 :  0.9035087719298246

'''
