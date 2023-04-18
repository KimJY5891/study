# 그리드서치의 문제점 : 다 돌린다는것 
# 과연 다 돌려도 되는걸까 
# 훈련의 전체의 max_resourse = 최대 리소스 = 최대 자원 = 1437 
# 훈련의 전체의 min_resourse = 최대 리소스 = 최대 자원 = 100
# iter = 반복 0 - 1번째, 1-2번째 
# n_resorce : 100 = 데이터 백개로 훈련시키겠다. 
# n_possivle_iterations : 3- 가능한 반복
# factor :  3 - 요인 #  디폴트 : 3
# iter : 0
# n_candiates : 52 
# n_resoureces : 100
# iter : 1 
# n_candiates : 18 -> 52개 중에서 3분의1인, 상위 18개를 사용하겟다. 
# n_resoureces : 300 - > 100*3
# iter : 2
# n_candiates : 6 -> 18개 중에서 3분의1인, 상위 6개를 사용하겟다. 
# n_resoureces : 900 - > 300*3

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, HalvingGridSearchCV
from sklearn.metrics import accuracy_score # 평가지표 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
import time

# 1. 데이터 
# x,y = load_iris(return_X_y=True)
x,y = load_digits(return_X_y=True)

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
# model = GridSearchCV(SVC(),
# model = RandomizedSearchCV(SVC(),
model = HalvingGridSearchCV(SVC(),      
                    # 380번 돌았다. 왜 그런 걸까? 
                    paramiters, # 52번돌림
                    #cv=kfold, # 5번돌림  
                    cv=5, 
                    verbose=1,
                    refit=True,
                    #n_iter=5, # 5 candiates 5개만 하겠다. - 디폴트 10 # 랜덤 서치놈의 파라미터라 안돼  
                    factor =3.2, 
                    #factor =2, # 디폴트 3
                    # factor =3 일 때 보다 iter가 0~3까지 
                    # 소수점 된다. 
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
# 로드디지트 그리드 서치걸린시간 : 12.61초
# 로드디지트 할빙 그리드 서치  걸린시간 : 5.13초
print(x.shape, x_train.shape)


'''

y_pred = model.predict(x_test)
print('acc_score : ',accuracy_score(y_test,y_pred))

y_pred_best= model.best_estimator_.predict(x_test)
print('최적의 튠  acc : ',accuracy_score(y_test,y_pred_best))

#print('결과 : ', model.cv_results_)
print('결과 : ', pd.DataFrame(model.cv_results_))
print('결과 : ', pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=False))
print('결과 : ', pd.DataFrame(model.cv_results_).columns)
path= './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=False)\
    .to_csv(path+'m12_RandomizedSearch1.csv')
'''
