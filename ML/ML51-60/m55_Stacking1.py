import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.ensemble import RandomForestClassifier


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
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt =DecisionTreeClassifier()
#  이 코딩에서는 
# model= DecisionTreeClassifier()
# model = VotingClassifier(
model = StackingClassifier(
    estimators=[('LR',lr),('KNN',knn),('DT',dt)], # # 이제 모델 작성  평가
    # final_estimator=DecisionTreeClassifier() #
    # final_estimator=LogisticRegression() # 셋중 하나가 디폴트 일것 
    final_estimator=VotingClassifier() # 이런식으로 스태킹안에 보팅할 수 있다. 
)
# 스테킹 최대 문제점 
# : 계산한걸 또 계산해서 과적합 
# 프레딕하면서 x_test로 만든 데이터로 훈련 시킨 것 
# 데이터를 나누지 않는 경우 리키지는 아니고 과적합이다. 
# 1) train을 2개로 나누기
# 2) 채우기
# 

# 3. 훈련 
model.fit(x_train,y_train)
print(model)
print(model.__class__.__name__) # 모델 이름만 나옴 
# 4. 평가,예측
y_pred = model.predict(x_test)
print('model.score: ',model.score(x_test,y_test))
print('acc : ',accuracy_score(y_test,y_pred))

classifiers = [lr,knn,dt]
for model2 in classifiers :
    model2.fit(x_train,y_train)
    y_pred = model2.predict(x_test)
    score2= accuracy_score(y_test,y_pred)
    class_name = model2.__class__.__name__
    print("{0}정확도 : {1:.4f}".format(class_name,score2)) # 0 에 클래스 네임 , 스코어2는 1:4f 이고 소수 4번째 짜리 까지 
