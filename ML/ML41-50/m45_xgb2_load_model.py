# 오류남 수정요망
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


# 2. 모델 # 3. 훈련 - 모델 불러오기
# 로드 모델 
path = './_save/pickle_test/'
model = XGBClassifier()
# model.load_model(path + 'm45_xgb1_save_model.dat')
# 픽클이랑 잡립과 다르게 모델을 먼저 정의해줘야한다. 
# model.load_model(path + 'm43_pickle1_save.dat')
model.load_model(path + 'm44_joblib1_save.dat')
# 위에 두 개 안 된다. 


# 4. 평가 
print('===========================================================')
result = model.score(x_test,y_test)
print("최종점수 : ", result) # 스코어가 엑큐러시라는 의미

y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print('acc : ', acc)
'''
최종점수 :  0.9122807017543859
acc :  0.9122807017543859
'''
