import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score # 평가지표 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터 
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    #stratify=y
)
n_split = 5
kfold= StratifiedKFold(n_splits=n_split, shuffle=True, random_state=337)

# paramiters = [
#     {"n_estimators" : [100,200]},
#     {"max_depth" : [6,8,10,12]},
#     {"min_samples_leaf" : [3,5,7,10]},
#     {"min_samples_split" : [2,3,5,10]},
#     {"n_jobs" : [-1]}    
# ]

paramiters = [
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"n_jobs" : [-1]},
]

#2. 모델

model = GridSearchCV(RandomForestClassifier(),
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
최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, min_samples_split=10,
                       n_jobs=-1)
최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 100, 'n_jobs': -1}
best_score_ :  0.9666666666666668
model.score :  0.9666666666666667
걸린 시간 :  91.0
acc_score :  0.9666666666666667
최적의 튠  acc :  0.9666666666666667
'''
'''
m05_kfold_01_iris.py
acc :  [1.         1.         0.93333333 0.9        1.        ]
cross_val_score 평균 :  0.9667
'''
