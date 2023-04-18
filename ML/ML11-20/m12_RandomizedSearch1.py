# 그리드서치와 랜덤서치 중에서  랜덤 서치가 가끔 더 잘나옴 
# 원리는 같다
# 그리드 서치는 모두 다 돌리는 것 
# 랜덤서치는  그 중에 랜덤하게 몇개만 뽑아서 만듬
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score # 평가지표 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
import time

# 1. 데이터 
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    #stratify=y
)
n_split = 5
kfold= StratifiedKFold(n_splits=n_split, shuffle=True, random_state=337)
# StratifiedKFold - 분류라서 사용 
paramiters = [
    {"C" : [1,10,100,1000],"kernel":['linear'],'degree':[3,4,5]}, # 12
    {"C" : [1,10,100],"kernel":['rbf','linear'],'gamma':[0.001,0.0001]}, #12 
    {"C" : [1,10,100,1000],"kernel":['sigmoid'],'gamma':[0.01,0.0001],'degree':[3,4]}, #24
    {"C" : [0.1,1],'gamma':[1,10]}, # 4
] # 총 52번


# 2. 모델
#model = GridSearchCV(SVC(),
model = RandomizedSearchCV(SVC(),
                     paramiters, # 52번돌림
                     #cv=kfold, # 5번돌림  
                     cv=4, 
                     verbose=1,
                     #n_iter=5, # 5 candiates 5개만 하겠다. - 디폴트 10 
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


# 우리가 왜 훈련 10번만 했는지 
# 랜덤 서치의 돌아가는 연산량 : 52개에 하나당 10개만 선택하겟다.
# 데이터 양 많은 대는 랜덤서치
# 데이터 양이 적으면 GridSearchCV
