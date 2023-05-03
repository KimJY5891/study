import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from catboost import CatBoostClassifier,CatBoostRegressor
import hyperopt
import pandas as pd
import numpy as np
print(hyperopt.__version__) # 0.2.7
from hyperopt import hp, fmin, tpe, Trials

# 1. 데이터 

x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337,train_size=0.8,shuffle=True,
    stratify=y
)
scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
search_space = { 
    'learning_rate' : hp.uniform('learning_rate', 0.001,1.0), # hp.uniform : 정규 분포 , 중앙 수치가 0.5로 예측 가운데로 갈수록 분포가 많다. 
    "depth" : hp.quniform('depth',3,16,1),  
    "num_leaves" : hp.quniform('num_leaves',24,64,1.0), # 간격이 1이기 때문에 round 처리 안해도 된다.
    "one_hot_max_size" : hp.quniform('one_hot_max_size',24,64,1.0), # 간격이 1이기 때문에 round 처리 안해도 된다.
    "min_data_in_leaf" : hp.quniform('min_data_in_leaf',10,100,1), # 정수 형태 
    "bagging_temperature" : hp.uniform('bagging_temperature',0.5,1), # 정수 형태 
    "random_strength" : hp.uniform('random_strength',0.5,1), # 정수 형태 
    "12_leaf_reg" : hp.uniform('12_leaf_reg',0.001,10),
}

import time
def cat_hamsu(search_space) : 
    params = {
        'iterations' : 10,
        "depth" : search_space['depth'],  
        "num_leaves" : int(search_space['num_leaves']), # 간격이 1이기 때문에 round 처리 안해도 된다.
        "one_hot_max_size" : search_space['one_hot_max_size'], # 간격이 1이기 때문에 round 처리 안해도 된다.
        "min_data_in_leaf" : search_space['min_data_in_leaf'], # 정수 형태 
        "bagging_temperature" : search_space['bagging_temperature'], # 정수 형태 
        "random_strength" : int(search_space['random_strength',0.5,1]), # 정수 형태 
        "12_leaf_reg" : int(search_space['12_leaf_reg']),
        'task_type' : 'CPU',
        'logging_level' : 'Silent'
    }
    model = CatBoostClassifier(
        **params,
        verbose=0
        )
    
    # 3. 훈련
    model.fit(x_train,y_train)

    # 4. 평가, 예측 
    y_pred = model.predict(x_test)
    result_mse = mean_squared_error(y_test,y_pred)
    return result_mse
start = time.time()
trials_val = Trials()
best = fmin(
    fn=cat_hamsu,
    space=search_space,
    algo= tpe.suggest,
    max_evals = 50,
    trials = trials_val,
    rstate=np.random.default_rng(seed = 10)
)
print('best : ',best)
'''


'''
