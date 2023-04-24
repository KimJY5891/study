import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier #  나무 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # 숲
from xgboost import XGBClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt 
scaler_list = [MinMaxScaler(),StandardScaler(),MaxAbsScaler(),RobustScaler()]
model_list = [DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)
# scaler = StandardScaler()
# x_trian = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
result = model.score(x_test,y_test)
print("model.score : ",result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)

from xgboost.plotting import plot_importance
plot_importance(model)
#모델의 가중치값이 훈련시켜놔서 다 들어있음
plt.show()
# 랜덤 포레스트는 안된다. 
# f score =/= f1 score
# f1 ,f2  => 피쳐(특성,열) 순서 
