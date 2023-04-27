# 가중치도 하나의 데이터이기 대문에 저장하면 된다. 
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


# 2. 모델 # 3. 훈련 - 피클 불러오기
import pickle
path = './_save/pickle_test/'
model = pickle.load(open(path+'m43_pickle1_save.dat','rb'))
# rb : read 바이너리 
import joblib
model = joblib.load(open(path+'m44_joblib1_save.dat'))

# 원래 둘 다 일반 데이터 저장용도 인데, 가중치도 데이터니까 활용가능! 
# 판다스 같은거를 이걸로 세이브 가능 
# to넘파이로 저장하는 경우도 있지만 
# 판다스 형태로 최종 정제를 한 다음 저장해야할 joblib에 데이터 프레임 모양으로 저장하면 된다.




# 4. 평가 - x 
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
