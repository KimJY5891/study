import pandas as pd
import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, HalvingGridSearchCV
from sklearn.metrics import accuracy_score,r2_score 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
import time

# 1. 데이터 
# x,y = load_iris(return_X_y=True)
x,y = load_digits(return_X_y=True)
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    #stratify=y
)
n_split = 5
kfold= StratifiedKFold(n_splits=n_split, shuffle=True, random_state=1234)
# StratifiedKFold - 분류라서 사용 
paramiters = [
    {"C" : [1,10,100,1000],"kernel":['linear'],'degree':[3,4,5]}, # 12
    {"C" : [1,10,100],"kernel":['rbf','linear'],'gamma':[0.001,0.0001]}, #12 
    {"C" : [1,10,100,1000],"kernel":['sigmoid'],'gamma':[0.01,0.0001],'degree':[3,4]}, #24
    {"C" : [0.1,1],'gamma':[1,10]}, # 4
] # 총 52번

# 2. 모델
model = HalvingGridSearchCV(SVC(),      
                    paramiters, # 52번돌림
                    #cv=kfold, # 5번돌림  
                    cv=5, 
                    verbose=1,
                    refit=True,
                    #n_iter=5, # 5 candiates 5개만 하겠다. - 디폴트 10 # 랜덤 서치놈의 파라미터라 안돼  
                    factor =3.3,
                    n_jobs=-1) # 총260번 돌림

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_) 
print("최적의 파라미터 : ",model.best_params_) 
print("best_score_ : ",model.best_score_) # train의 베스트 스코어 
print("model.score : ",model.score(x_test,y_test)) # test의 베스트 스코어 
end_time = time.time()
print('걸린 시간 : ',np.round(end_time-start_time))
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

'''
해빙
Fitting 5 folds for each of 5 candidates, totalling 25 fits
최적의 매개변수 :  SVC(C=10, gamma=0.001, kernel='linear')
최적의 파라미터 :  {'C': 10, 'gamma': 0.001, 'kernel': 'linear'}
best_score_ :  0.9627906976744185
model.score :  0.9298245614035088
걸린 시간 :  25.0
'''
# 결론 : 해빙 승
