# hyperopt은 최소값 찾기
# 베이지안 옵티마이제이션은 최대 값을 찾는것 
# pip install hyperopt
import hyperopt
import pandas as pd
import numpy as np
print(hyperopt.__version__) # 0.2.7
from hyperopt import hp, fmin, tpe, Trials# 통상 이렇게 쓴다.

param_bounds = {
    'x1':(-1,5),
    'x2':(0,4),
                } # 딕셔너리 

search_space = {
    'x1' : hp.quniform('x1',-10,10,0.5), # 소수도 가능하다. 
    'x2' : hp.quniform('x2',-15,15,1)
    # hp.quniform(label,low,high,q) 
    # low : 최소값 
    # high : 최대값 
    # q :간격, 1일경우 15,16,17 이런식 
} # 파라미터를 모아둔 것 
print(search_space)

def objectitve_func(search_space) :
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2-20*x2
    # x1 : 0
    # x2 : 15 일때 최소값 
    return return_value # 모델예씨 

trials_val = Trials()
# Trials은 히스토리와 같은 부분

# fmin : 최소값을 찾는 함수 
best = fmin(
    fn = objectitve_func, # 목적함수 
    space = search_space, # 파라미터 
    algo = tpe.suggest, # 디폴트 알고리즘 선택하는 파라미터 
    max_evals=20,# n_iter과 같은넘 10번 돌리겠다. 
    trials=trials_val,
    # rstate= np.random.default_rng(seed=10) #랜덤 스테이트 빼면 고정안되서 매번 바뀐다. 
)
print('best : ',best)
print('trials_val.results : ',trials_val.results) # 
print('trials_val.vals : ',trials_val.vals) # 내역이 나온다. 
#  {'loss': 80.0, 'status': 'ok'}
'''
성능이 더 좋다고 소문이 있다.
'''

###### 트라이얼 발 판다스에 데이터 프레임에 넣기 ########## 

trials_val_vals = {
    'x1': [6.0, -7.0, -5.0, -10.0, 4.0, 6.0, 8.0, -7.0, -6.0, 5.0, -1.0, 4.0, -10.0, -4.0, -7.0, -8.0, -6.0, -10.0, 
5.0, 7.0], 
    'x2': [-7.0, -11.0, -7.0, -10.0, 13.0, 6.0, 5.0, -7.0, 6.0, -6.0, -10.0, -9.0, -14.0, -3.0, -14.0, 11.0, -5.0, 12.0, -6.0, -13.0]}


trials_val_vals_col = ['x1','x2']

trials_val_vals = pd.DataFrame(trials_val_vals,columns=trials_val_vals_col,
                               # index=False
                               )

print(trials_val_vals)
results = [aaa['loss'] for aaa in trials_val.results]
# 위 아래 동일
for aaa in trials_val.results : 
    results.append(aaa['loss'])
    
# 결과값 넣기 
df = pd.DataFrame({'x1' : trials_val.vals['x1'],
                   'x2' : trials_val.vals['x2'],
                   'results' : results
                   })
print(df)
