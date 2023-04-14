import numpy as np
from sklearn.datasets import load_wine, load_digits, load_breast_cancer, fetch_covtype, load_iris, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

datasets_name = ['load_wine', 'load_digits', 'load_breast_cancer', 'fetch_covtype', 'fetch_covtype', 'load_diabetes']

for i in range(len(datasets_name)):
    # 1. 데이터 
    x, y = eval(datasets_name[i])(return_X_y=True)
    print(x.shape, y.shape)
    # 2. 모델 
    model = DecisionTreeRegressor()
    # 3. 컴파일, 훈련 
    model.fit(x, y) # 핏에 컴파일 포함
    # 4. 평가 예측
    results = model.score(x, y)
    print(datasets_name[i], ':', results)
