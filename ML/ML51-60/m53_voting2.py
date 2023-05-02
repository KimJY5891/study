# 
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor # 원래 회귀 분류 나누지 않았는데, 사이킷런 문법 따라하면서 회귀 분류로 나뉜다. 
from lightgbm import LGBMRegressor # 원래 회귀 분류 나누지 않았는데, 사이킷런 문법 따라하면서 회귀 분류로 나뉜다. 
# 트리구조 애들이 깊어(맥스뎁스) 질 수록  연산량도 많아지고 과적합 가능성도 있다. 잘 조절해야하는 영향력있는 파라미터 
# 샘플 - 드랍아웃과 비슷하다. 
# 찾아보기 
# 라이트 지비엠은  유효한 쪽으로 치우쳐있다.
# 다른 트리 기반 모델과 비교하여, 효율적인 메모리 사용과 빠른 속도를 제공하는 것이 특징입
# 대신에 맥스뎁스가 더 깊어진다.
from catboost import CatBoostRegressor


# 1. 데이터
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337,train_size=0.8,shuffle=True,
    # stratify=y
)
scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
xg = XGBRegressor()
lg = LGBMRegressor()
cb = CatBoostRegressor(verbose=0)

# model= DecisionTreeClassifier()
model = VotingRegressor(
    estimators=[('XG',xg),('LG',lg),('CB',cb)], 
    # voting = 'hard', 
    # voting = 'soft', # regressor에서 voting 하이퍼 파라미터가 안먹힌다. 
    # voting 이 안먹히는 이유 1과 0 같이 저수로 나뉘누는 것이 아니기 때문선택하는게 아니라 선택하는게 아니라 보팅 파라미터가 어렵다. 
    # 대신평균낸다. 
    # 배깅과의 차이점 - 배깅은 하나의 모델 보팅은 여러가지 모델 
    # 배깅과 보팅의 차이점 정리하기 
)
# 3. 훈련 
model.fit(x_train,y_train)

# 4. 평가,예측
y_pred = model.predict(x_test)
print('model.score: ',model.score(x_test,y_test))
print('r2_score : ',r2_score(y_test,y_pred))

regressor = [xg,lg,cb]
for model2 in regressor :
    model2.fit(x_train,y_train)
    y_pred = model2.predict(x_test)
    score2= r2_score(y_test,y_pred)
    class_name = model2.__class__.__name__
    print("{0} r2 : {1:.4f}".format(class_name,score2))
'''
acc :  0.9385964912280702
LogisticRegression정확도 : 0.9561
KNeighborsClassifier정확도 : 0.9737
DecisionTreeClassifier정확도 : 0.8860
단일 모델 보다 좋을 수도 있고 나쁠수도 있다. 알아서 잘 사용해야한다. 
'''
    



'''
랜덤포레스트
model.score:  0.9385964912280702
acc :  0.9385964912280702
'''
'''
배깅 부트스트랩 펄스
model.score:  0.9385964912280702
acc :  0.9385964912280702
'''
'''
배깅 부트스트랩 트루
model.score:  0.9385964912280702
acc :  0.9385964912280702
'''
'''
하드 보팅
model.score:  0.9736842105263158
acc :  0.9736842105263158   
'''
'''
soft 보팅
model.score:  0.9385964912280702
acc :  0.9385964912280702 
'''
'''
배깅 2개 랜덤포레스트 트루 
model.score:  0.9385964912280702
acc :  0.9385964912280702
'''
