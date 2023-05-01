import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이텉
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337,train_size=0.8,shuffle=True,
    stratify=y
)
scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # 리그레서이지만 분류이다. 

# model= DecisionTreeClassifier()
# model= RandomForestClassifier() # DecisionTree가 모여있는게 랜덤 포레스트 이다. 
model= BaggingClassifier(
    LogisticRegression(), 
    n_estimators=100, # 안에 작성한 모델을 10 번 하느 ㄴ것 
    n_jobs = -1, 
    random_state=337,
    bootstrap= True, 
    # 중복허용 데이터 샘플에서 똑같은 두 개 세개 나올 수도 있다.
    # 그렇게 작업했을 경우 성능이 오히려 좋아질 수도 잇다.
    # 통상적으로 성능이 더 좋다. 
    # bootstrap= False, #
    ) # DecisionTree가 모여있는게 랜덤 포레스트이다. 
# 배깅 안에 배깅 넣어도 가능하다.

# 3. 훈련 
model.fit(x_train,y_train)

# 4. 평가,예측
y_pred = model.predict(x_test)
print('model.score: ',model.score(x_test,y_test))
print('acc : ',accuracy_score(y_test,y_pred))
