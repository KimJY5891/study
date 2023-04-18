import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import pprint


# 1. 데이터 
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    #stratify=y
)

n_split = 5
kfold= StratifiedKFold(n_splits=n_split, shuffle=True, random_state=337)

paramiters = [
    {"C" : [1,10,100,1000],"kernel":['linear'],'degree':[3,4,5]}, # 12
    {"C" : [1,10,100],"kernel":['rbf','linear'],'gamma':[0.001,0.0001]}, #12 
    {"C" : [1,10,100,1000],"kernel":['sigmoid'],'gamma':[0.01,0.0001],'degree':[3,4]}, #24
    {"C" : [0.1,1],'gamma':[1,10]}, # 4
] # 총 52번

# 2. 모델

model = GridSearchCV(SVC(),
                     paramiters, # 52번돌림
                     # cv=kfold, # 5번돌림  
                     cv=5,
                     verbose=1,
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
# 최적의 튠  acc :  1.0

###########################################################################
#print('결과 : ', model.cv_results_)
print('결과 : ', pd.DataFrame(model.cv_results_))
print('결과 : ', pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=False))
print('결과 : ', pd.DataFrame(model.cv_results_).columns)
path= './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=False)\
    .to_csv(path+'m10_GridSearch3_cv_results_.csv')

# 설명 
'''
각 52개 파라미터 52번 돈 것의 결과
판다스로 행렬해줘서 만들기
판다스로 만드는 건 데이터 모양만들어서 하기
 {'mean_fit_time': array([0.00019989, 0.00122347, 0.        , 0.00312471, 0.0031249 ,
       0.        , 0.        , 0.00312433, 0.        , 0.        ,
       0.00312486, 0.00312486, 0.        , 0.        , 0.00312505,
       0.        , 0.00312452, 0.        , 0.        , 0.        ,
       0.        , 0.00312476, 0.        , 0.00312476, 0.        ,
       0.        , 0.        , 0.00312505, 0.00339541, 0.00474358,
       0.00039968, 0.0001997 , 0.00039983, 0.        , 0.        ,
       0.00287929, 0.        , 0.        , 0.00099931, 0.00059962,
       0.00039964, 0.00199885, 0.00129199, 0.        ]), 'std_fit_time': array([3.99780273e-04, 3.90488514e-04, 0.00000000e+00, 6.24942780e-03,
       6.24980927e-03, 0.00000000e+00, 0.00000000e+00, 6.24866486e-03,
       0.00000000e+00, 0.00000000e+00, 6.24971390e-03, 6.24971390e-03,
       0.00000000e+00, 0.00000000e+00, 6.25009537e-03, 0.00000000e+00,
       6.24904633e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 6.24952316e-03, 0.00000000e+00, 6.24952316e-03,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.25009537e-03,
       6.79082870e-03, 6.06812075e-03, 4.89512132e-04, 3.99398804e-04,
       4.89687292e-04, 0.00000000e+00, 0.00000000e+00, 4.27683230e-03,
       0.00000000e+00, 0.00000000e+00, 1.90734863e-07, 4.89590007e-04,
       4.89453644e-04, 1.09458223e-03, 1.20450315e-03, 0.00000000e+00]), 'mean_score_time': array([0.        , 0.00059967, 0.        , 0.        , 0.        ,
       0.        , 0.00312519, 0.        , 0.        , 0.        ,
       0.00312486, 0.        , 0.00312505, 0.        , 0.00312505,
       0.        , 0.        , 0.        , 0.00312452, 0.        ,
       0.00312452, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.00312505, 0.        , 0.00079956,
       0.00019994, 0.00227957, 0.00031648, 0.        , 0.        ,
       0.00031643, 0.        , 0.00314469, 0.        , 0.00059967,
       0.00019989, 0.00059967, 0.        , 0.        ]), 'std_score_time': array([0.        , 0.00048963, 0.        
, 0.        , 0.        ,
       0.        , 0.00625038, 0.        , 0.        , 0.        ,
       0.00624971, 0.        , 0.0062501 , 0.        , 0.0062501 ,
       0.        , 0.        , 0.        , 0.00624905, 0.        ,
       0.00624905, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.0062501 , 0.        , 0.00074792,
       0.00039988, 0.00455914, 0.00040937, 0.        , 0.        ,
       0.00040929, 0.        , 0.00628939, 0.        , 0.00048963,
       0.00039978, 0.00048963, 0.        , 0.        ]),                
'''
print('결과 : ', pd.DataFrame(model.cv_results_))
# 판다스에서의 모양 : 리스트와 데이터 프레임
'''
판다스 
결과 :      mean_fit_time  std_fit_time  mean_score_time  ...  mean_test_score std_test_score rank_test_score
0        0.000400  7.996559e-04         0.003146  ...         0.991667       0.016667               1
1        0.003146  6.292534e-03         0.000200  ...         0.991667       0.016667               1
2        0.001199  3.997566e-04         0.000400  ...         0.991667       0.016667               1
3        0.000207  4.141808e-04         0.000000  ...         0.966667       0.048591               7
4        0.001001  2.949460e-06         0.000400  ...         0.966667       0.048591               7
5        0.000198  3.967285e-04         0.000227  ...         0.966667       0.048591               7
6        0.003125  6.249905e-03         0.000000  ...         0.950000       0.061237              13
...
42       0.000000  0.000000e+00         0.000000  ...         0.975000       0.050000               6
43       0.000000  0.000000e+00         0.000000  ...         0.941667       0.042492              24
[52 rows x 17 columns]???
'''

print('결과 : ', pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=False))
print('결과 : ', pd.DataFrame(model.cv_results_).columns)
# sort_values('rank_test_score') : rank_test_score 이것의 스코어 순으로 하겠다. 
# ascending=True 디폴트

path= './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=False)\
    .to_csv(path+'m10_GridSearch3_cv_results_.csv')
# 이런 식으로 했을 때 \로 줄바꿈할 수 있다. 두줄이지만 한 줄의 의미가 있다. 

'''
결과 :      mean_fit_time  std_fit_time  mean_score_time  ...  mean_test_score std_test_score rank_test_score
25       0.000000      0.000000         0.000000  ...         0.366667       0.016667              37
41       0.001647      0.002422         0.000400  ...         0.366667       0.016667              37
....
4        0.000000      0.000000         0.000000  ...         0.966667       0.048591               7
19       0.000000      0.000000         0.000000  ...         0.966667       0.048591               7
42       0.000218      0.000436         0.000000  ...         0.975000       0.050000               6
13       0.000000      0.000000         0.000000  ...         0.991667       0.016667               1
1        0.000000      0.000000         0.003125  ...         0.991667       0.016667               1
15       0.000000      0.000000         0.003125  ...         0.991667       0.016667               1
2        0.003125      0.006250         0.000000  ...         0.991667       0.016667               1
0        0.000000      0.000000         0.000000  ...         0.991667       0.016667               1

'''
'''
결과 :  Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_C', 'param_degree', 'param_kernel', 'param_gamma', 'params',
       
    아래는 크로스발리데이션에 대한 자른 순서에 대한 스코어 
    'split0_test_score', 'split1_test_score', 'split2_test_score','split3_test_score', 'split4_test_score',   
       
       'mean_test_score',
       'std_test_score', 'rank_test_score'],
      dtype='object')
'''

