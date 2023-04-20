import numpy as np
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score # 평가지표 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터 
x,y = load_diabetes(return_X_y=True)

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

model = RandomizedSearchCV(RandomForestRegressor(),
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
print('acc_score : ',r2_score(y_test,y_pred))
y_pred_best= model.best_estimator_.predict(x_test)
print('최적의 튠  acc : ',r2_score(y_test,y_pred_best))
'''
#그리드 
Fitting 5 folds for each of 274 candidates, totalling 1370 fits
최적의 매개변수 :  RandomForestRegressor(max_depth=8, min_samples_leaf=7, min_samples_split=5,
                      n_estimators=200, n_jobs=-1)
최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 7, 'min_samples_split': 5, 'n_estimators': 200, 'n_jobs': -1}
best_score_ :  0.45537595994434704
model.score :  0.44956850980869556
걸린 시간 :  193.0
'''
'''
# 랜덤
Fitting 5 folds for each of 10 candidates, totalling 50 
fits
최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=7, n_jobs=-1)
최적의 파라미터 :  {'n_jobs': -1, 'n_estimators': 100, 'min_samples_leaf': 7}
best_score_ :  0.44928988264306263
model.score :  0.45582106096300345
걸린 시간 :  6.0
acc_score :  0.45582106096300345
최적의 튠  acc :  0.45582106096300357
'''

# 그리드 승
