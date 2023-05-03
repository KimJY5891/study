import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,fetch_california_housing, load_iris, load_wine,load_digits,fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score, r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor, XGBClassifier 
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor, CatBoostClassifier

import hyperopt
import pandas as pd
import numpy as np
print(hyperopt.__version__) # 0.2.7
from hyperopt import hp, fmin, tpe, Trials

#1. 데이터

x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337, train_size=0.8, stratify=y
)
scaler = RobustScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# 2. 모델

search_space = { # 범위라서 여기에 int나 round ㄴㄴ 
    "max_depth" : hp.quniform('max_depth',3,16,1),  
    'learning_rate' : hp.uniform('learning_rate', 0.001,1.0), # hp.uniform : 정규 분포 , 중앙 수치가 0.5로 예측 가운데로 갈수록 분포가 많다. 
    "num_leaves" : hp.quniform('num_leaves',24,64,1.0), # 간격이 1이기 때문에 round 처리 안해도 된다.
    "min_child_weight" : hp.uniform('min_child_weight',10,200), # 정수 형태 
    # "min_child_samples" : hp.quniform('min_child_samples',1,50),
    "subsample" : hp.uniform('subsample',0.5,1),
    # "colsample_bytree" : hp.uniform('colsample_bytree',0.5,1),
    # "max_bin" : hp.quniform('max_bin',8,500), # 값이 10이상 줘야한다.  
    # "reg_alpha":hp.uniform('reg_alpha',0.001,10), 
    # "reg_lambda":hp.uniform('reg_lambda',0.01,50)
} # 파라미터 범위 
# 이 값이 매개변수를 타고 함수 속으로 들어간다. 
'''
hp.quniform은 이산형 하이퍼파라미터를 샘플링할 때,
hp.uniform은 연속형 하이퍼파라미터를 샘플링할 때 사용됩니다.

'''

#  x의 값이 큰걸 로그 변환으로 
# hp.quniform(label,low,high,q) : 최소부터 최대까지 q 간격
# hp.uniform(label,low,high,q) : 최소부터 최대까지 정규분표의 간격
# hp.loguniform(label,low, high) : exp(uniform(low,high))  값 반환 / 이거 역시 정규 분포 
# 프레딕트할 때는 지수 변환해서 
# 
import time

def lgbm_hamsu(search_space) : 
    params = {
        # 'n_estimate' : 1000,
        'learning_rate' :search_space['learning_rate'], # 중요 
        "max_depth": int(search_space['max_depth']), # 무조건 정수형
        "num_leaves" : int(search_space['num_leaves']),# 정수형
        "min_child_weight" : int(search_space['min_child_weight']), 
        # "min_child_samples" : int(round(min_child_samples)),
        # 그냥 int보다 round를 추가의 이점은 18.68 값에 대해서은 19를 사용하고 싶어질 때 round사용할 수 있다. 
        "subsample" : max(min(search_space['subsample'],1),0), # 드랍아웃 똑같다고하면 틀리고, 비슷하다고 생각해야한다. 
        # "colsample_bytree" : colsample_bytree, 
        # "max_bin" : max(int(round(max_bin)),10),# 여러 값일 경우 10이상인 값을 내놔라  # 무조건 10 이상 
        # "reg_alpha" : max(reg_alpha,0), # 무조건 양수만 
        # "reg_lambda" :reg_lambda
    }
    model= LGBMRegressor(**params)         
    model.fit(
        x_train,y_train,
        eval_set = [(x_train,y_train),(x_test,y_test)],
        eval_metric='rmse',
        verbose=0,
        early_stopping_rounds=50
        )
    y_pred = model.predict(x_test)
    result_mse = mean_squared_error(y_test,y_pred)
    #  그대로 받아들일 경우 
    return result_mse

start= time.time()

trials_val = Trials()
best = fmin(
    fn=lgbm_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals= 50,
    trials=trials_val,
    rstate=np.random.default_rng(seed=10)
)
print('best : ',best)
'''
best :  {'learning_rate': 0.5228252684055418, 'max_depth': 9.0,
'min_child_weight': 10.088498014452998, 'num_leaves': 60.0,
'subsample': 0.9481142745011609}
result의 최소값
'''

# print('trials_val.results : ',trials_val.results) # 
# print('trials_val.vals : ',trials_val.vals) 

end= time.time()
print('시간 : ',round(end-start,3))
### 결과는 판다스 데이터 프레임형태로 빼자 !! ###
## result 컬럼에 최소값이 있는 행을 출력 
