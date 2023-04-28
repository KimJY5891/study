
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score

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

#2. 모델 
model = XGBClassifier(**parameter)

# 3. 훈련
hist = model.fit(
    x_train,y_train,
    eval_set =[(x_train,y_train),(x_test,y_test)], 
    early_stopping_rounds = 10, 
    verbose= True 
)

# 4. 평가, 예측 
print('===========================================================')
result = model.score(x_test,y_test)
print("최종점수 : ", result) # 스코어가 엑큐러시라는 의미

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('acc : ', acc)

#######################################################################

path = './_save/pickle_test/'
# 저장 
model.save_model(path+ 'm45_xgb1_save_model.dat')
'''
최종점수 :  0.9122807017543859
acc :  0.9122807017543859
'''
