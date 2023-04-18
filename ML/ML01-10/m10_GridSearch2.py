import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score # 평가지표 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
# cv : 크로스 발리데이션
# 파라미터 전체적으로 돌리는것 + 크로스 발리데이션도 같이해준다. 
# 그리드 서치 하는 이유 최고의 파라미터가 뭔지 알아내는 것 

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
# 키밸류 형태로 받기위해서 리스트안에 딕셔너리 형태 변환

# 2. 모델 
model = GridSearchCV(SVC(),
                     paramiters, # 52번돌림
                     #cv=kfold, # 5번돌림  
                     cv=5, # 더 잘 나온 이유 : cv의 디폴트가 StratifiedKFold이다. 그래서 파람 조절을  cv = 5를 기본으로 줬을 더 나을 수 잇다. 회귀라면 kfold
                     # 
                     verbose=1,
                     refit=True,
                     n_jobs=-1) # 총260번 돌림

# 매개변수 = 파라미터

# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_) # 
print("최적의 파라미터 : ",model.best_params_) #
print("best_score_ : ",model.best_score_) # train의 베스트 스코어 
print("model.score : ",model.score(x_test,y_test)) # test의 베스트 스코어 
end_time = time.time()
print('걸린 시간 : ',np.round(end_time-start_time))
#Fitting 5 folds for each of 40 candidates, totalling 200 fits
# 40 candidates : 후보자 = 파라미터
# 보이는 이유  verbose=1,
