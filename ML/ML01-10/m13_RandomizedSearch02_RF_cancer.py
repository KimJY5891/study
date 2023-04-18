#그리드 서치와 랜덤서치 비교 
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score # 평가지표 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

import time

# 1. 데이터 
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    #stratify=y
)
n_split = 5
kfold= StratifiedKFold(n_splits=n_split, shuffle=True, random_state=337)

paramiters = [
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"n_jobs" : [-1]},
]

#2. 모델

model = RandomizedSearchCV(RandomForestClassifier(),
                     paramiters,
                     cv=5,
                     verbose=1,
                     n_jobs=-1)

# 3. 컴파일, 훈련
start_time = time.time()

model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_) 
print("최적의 파라미터 : ",model.best_params_)
print("best_score_ : ",model.best_score_) # train의 베스트 스코어 
print("model.score : ",model.score(x_test,y_test)) # test의 베스트 스코어 

end_time = time.time()
print('걸린 시간 : ',np.round(end_time-start_time))

y_pred = model.predict(x_test)
print('acc_score : ',accuracy_score(y_test,y_pred))
y_pred_best= model.best_estimator_.predict(x_test)
print('최적의 튠  acc : ',accuracy_score(y_test,y_pred_best))
'''
# 그리드 서치
최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_leaf=3, n_jobs=-1)
최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1}
best_score_ :  0.956043956043956
model.score :  0.956140350877193
걸린 시간 :  105.0
acc_score :  0.956140350877193
최적의 튠  acc :  0.956140350877193
'''
'''
# 랜덤 서치
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3, min_samples_split=3,
                       n_estimators=200, n_jobs=-1)
최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 200, 'min_samples_split': 3, 'min_samples_leaf': 3, 'max_depth': 12}
best_score_ :  0.9516483516483516
model.score :  0.956140350877193
걸린 시간 :  7.0
acc_score :  0.956140350877193
최적의 튠  acc :  0.956140350877193
'''
