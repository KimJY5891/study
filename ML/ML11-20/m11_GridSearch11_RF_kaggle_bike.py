import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score,r2_score # 평가지표 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터
path = "./_data/kaggle_bike/"
path_save = "./_save/kaggle_bike/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################                                                                                 #
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)
####################################### 결측치 처리 #######################################                                                                                 #
x = train_csv.drop(['count'],axis=1)#
y = train_csv['count']
print("x.shape : ",x.shape)#(652, 8)
print("y.shape : ",y.shape)#(652, )
x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=123,test_size=0.2
)
n_splits = 5
kfold = KFold(
    n_splits = n_splits,
    shuffle=True,  
    random_state=123
)
paramiters = [
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10],"n_jobs" : [-1]},
    {"n_estimators" : [100,200],"n_jobs" : [-1]},
]

#2. 모델

model = GridSearchCV(RandomForestRegressor(),
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
print('r2_score : ',r2_score(y_test,y_pred))
y_pred_best= model.best_estimator_.predict(x_test)
print('최적의 튠  acc : ',r2_score(y_test,y_pred_best))
'''
#그리드 
Fitting 5 folds for each of 274 candidates, totalling 1370 fits
최적의 매개변수 :  RandomForestRegressor(n_jobs=-1)
최적의 파라미터 :  {'n_estimators': 100, 'n_jobs': -1}  
best_score_ :  0.9996098744305492
model.score :  0.9997977293289634
걸린 시간 :  635.0
r2_score :  0.9997977293289634
최적의 튠  acc :  0.9997977293289634
'''
'''
r2_score :  [0.99904053 0.99905398 0.99902148 0.99916546 0.99879329]
 cross_val_score 평균 :  0.999
'''

# 결론 : 무승부
