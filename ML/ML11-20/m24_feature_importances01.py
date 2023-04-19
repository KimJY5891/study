# 예) 500개 넘는 컬럼 중에서 성능에 영향을 끼치지 않을 수도 잇다. 
# 단점 : 자원낭비, 성능에 안좋은 영향 
# 안좋은 컬럼을 찾아내고자해서 feature_importances
# 트리계열에서만 제공해준다. 나머지 모델은 제공해주지 않는다. 
# 데이터 마다 다름
from sklearn.tree import DecisionTreeClassifier #  나무 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # 숲
from xgboost import XGBClassifier
# xg부스터 옛날 방식이 있고 사이킷런 방식이 있다. 
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# 1. 데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, random_state=337
)

scaler = MinMaxScaler()
x_trian = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
#model = DecisionTreeClassifier()
# model = RandomForestClassifier()
#model = GradientBoostingClassifier()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측

result = model.score(x_test,y_test)
print("model.score : ",result)

y_pred = model.predict(x_test)
print("y_pred : ",y_pred)

acc = accuracy_score(y_test,y_pred)
print("acc : ",acc)

print(model,":",model.feature_importances_)
# acc: 0.933333
#DecisionTreeClassifier() : [0.         0.05013578 0.9139125  0.03595171]
# acc값을 신뢰할 수 있다는 기준하에높을 수록 중요하다. 엑큐러시 93이 나오는 이 모델이 나오는 성능에서 3번째 컬럼이 가장 중요했다. 

# acc: 0.933333
# RandomForestClassifier() : [0.11631719 0.03602377 0.47782462 0.36983442]
# acc : 0.96666666
# GradientBoostingClassifier() : [0.00778376 0.01137374 0.72613721 0.25470528]
# acc : 0.96666666
# XGBClassifier : [0.01794496 0.01218657 0.8486943  0.12117416]
# 필요하다고 생각하는 컬럼만 사용해도 93프로이지만 오히려 잘 될 acc가 오를 수도 있다. 


##################################[실습] for문 사용해서 4개 돌리기 ######################################
scaler_list = [MinMaxScaler(),StandardScaler(),MaxAbsScaler(),RobustScaler() ]
model_list = [DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]
for i in scaler_list :
    # 1. 데이터
    x,y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, train_size=0.8, shuffle=True, random_state=337
    )
    scaler = i
    x_trian = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    for j in model_list :
        # 2. 모델
        model = j

        # 3. 훈련
        model.fit(x_train,y_train)

        # 4. 평가, 예측
        result = model.score(x_test,y_test)
        print("model.score : ",result)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test,y_pred)
        print("acc : ",acc)
        print("스케일러 : ",i)
        print(j,":",j.feature_importances_)
