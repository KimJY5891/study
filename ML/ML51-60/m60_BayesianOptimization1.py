param_bounds = {
    'x1':(-1,5),
    # 텍스트 형태랑 튜플형태로 넣어줘야한다.
    'x2':(0,4),
                } # 딕셔너리 
# bounds 범위 같은 느낌

def y_function(x1,x2) : 
    return -x1 **2 - (x2-2) **2 +10
# 엑큐러시나 mse,mae에 적용
# mse
# 마이너스가 먼저 계산됌
# 이 함수의 최대값 
# pip install bayesian-optimization
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization( # 최대 값을 찾는 것이다. 
    f = y_function,
    pbounds=param_bounds,
    random_state=337
)

optimizer.maximize(init_points=100,
                   n_iter=10 #실질적으로 22번 찾는다.
                   #
                   )
# maximize : 최대값 찾기 
print(optimizer.max)
# 분홍색 부분이 갱신된 것이다. 
# 
