from sklearn.tree import DecisionTreeClassifier #  나무 
from sklearn.tree import DecisionTreeRegressor #  나무 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # 숲
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, r2_score

scaler_list = [MinMaxScaler(),StandardScaler(),MaxAbsScaler(),RobustScaler()]
model_list = [DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

# 2. 모델
model01 = DecisionTreeClassifier()
model02 = RandomForestClassifier()
model03 = GradientBoostingClassifier()
model04 = XGBClassifier()

# 3. 훈련
model01.fit(x_train,y_train)
model02.fit(x_train,y_train)
model03.fit(x_train,y_train)
model04.fit(x_train,y_train)

import matplotlib.pyplot as plt 
def plot_feature_importances(model) : # 트리계열만 가능 
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1,n_features)
    plt.title(model)
plt.subplot(2,2,1)
# subplot(가로,세로,인덱스)
plot_feature_importances(model01)
plt.subplot(2,2,2)
plot_feature_importances(model02)
plt.subplot(2,2,3)
plot_feature_importances(model03)
plt.subplot(2,2,4)
plot_feature_importances(model04)
plt.show()      
