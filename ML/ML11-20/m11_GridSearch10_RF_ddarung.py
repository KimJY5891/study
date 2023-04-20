import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score,r2_score # 평가지표 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터
path = "./_data/dacon_ddarung/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################
train_csv = train_csv.dropna()

####################################### 결측치 처리 #######################################
x = train_csv.drop(['count'],axis=1)#
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=123,test_size=0.2,shuffle=True
)
n_splits = 5
kfold = KFold(n_splits = n_splits,
      shuffle=False,  
      #random_state=123
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
최적의 파라미터 :  {'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 100, 'n_jobs': -1}
best_score_ :  0.7560369775677639
model.score :  0.7846451857090704
걸린 시간 :  160.0
r2_score :  0.7846451857090704
최적의 튠  acc :  0.7846451857090704
'''
'''
r2_score :  [0.65606864 0.58509135 0.61342161 0.56255309 0.49143488]        
 cross_val_score 평균 :  0.5817
'''

# 결론 : 그리드 승
