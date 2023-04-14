import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore')
# 1. 데이터 
data_list= [load_iris(return_X_y=True),
      load_breast_cancer(return_X_y=True),
      load_wine(return_X_y=True),]
data_name_list = ['아이리스 : ', '브레스 캔서 : ','와인 : ']
model_list = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
model_name_list = ['LinearSVC : ','LogisticRegression : ','DecisionTreeClassifier : ','RandomForestClassifier :']
for i,value in enumerate(data_list) : 
    # 1. 데이터 
    x,y = value
    #print(x.shape, y.shape)
    print("================================")
    print(data_name_list[i])
    for j,value2 in enumerate(model_list) : 
        # 2. 모델 
        model= value2
        # 3. 컴파일, 훈련
        model.fit(x,y) # 핏에 컴파일 포함
        # 4. 평가 예측
        results = model.score(x,y)
        print(model_name_list[j],results)
# i와 j는 인덱스 수치
