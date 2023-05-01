# bagging -> 앙상블 모델 끼리 낸 가중치 값을 엔빵하는 것 
# boosting -> 가중치를 갱신시켜 다음 모델에 영향을 끼치는 것이 부스팅입니다. 
# voting -> 한 데이터로 각기 다른 모델로 0일 확률 1일 확률을 만들어낸다. 
# 서로 1일확률 0일 확률 이 다를 경우 투표를 한다.
# 1일 확률이 더 크다고 말하는 모델이 많으면 투표 하듯 1이라고 결정내린다. 
# soft voting : 하나의 데ㅣㅌ터를 각기 다른 모델로 0과 1에 대한 확률 을 내릴때, 0에 대한 모든 확률을 더해서 1.3, 1일 확률 을 모델이 내놓은 모든 확률 값을 더 해서 1.7일경우 
# 0일 확률 과 1일 확률을 더한 거끼리 싸워서 더 큰 쪽인 1.7로 결정을 내린다. 
# 스태킹 :  데이터를 모델로 .fit 하고 프레딕한 여러 프레딕의 결과를 모아서 다시 모델로 훈련 시키는 것
# 하지만 통상적으로 1.5% 성능향상이 있었다고 한다. 

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
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

# model= DecisionTreeClassifier()
 # ㅍDecisionTree가 모여있는게 랜덤 포레스트 이다. 
model= BaggingClassifier(
    # BaggingClassifier(
    RandomForestClassifier(), 
    n_estimators=100, # 안에 작성한 모델을 10 번 하느 ㄴ것 
    n_jobs = -1, 
    random_state=337,
    # bootstrap= True, 
    # 중복허용 데이터 샘플에서 똑같은 두 개 세개 나올 수도 있다.
    # 그렇게 작업했을 경우 성능이 오히려 좋아질 수도 잇다.
    # 통상적으로 성능이 더 좋다. 
    bootstrap= False, #
    # )
) # DecisionTree가 모여있는게 랜덤 포레스트이다. 
# 배깅 안에 배깅 넣어도 가능하다.
# model= RandomForestClassifier()
# 3. 훈련 
model.fit(x_train,y_train)

# 4. 평가,예측
y_pred = model.predict(x_test)
print('model.score: ',model.score(x_test,y_test))
print('acc : ',accuracy_score(y_test,y_pred))

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
배깅 2개 랜덤포레스트 트루 
model.score:  0.9385964912280702
acc :  0.9385964912280702
'''
