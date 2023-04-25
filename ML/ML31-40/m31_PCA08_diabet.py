### 실습 ### 
#for문 써서 한 번에 돌리기
# 기본 결과 : 
# 차원 1개 축소 :
# 차원 2개 축소 : 
 
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터 
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']
print(x.shape,y.shape) #(442, 10) (442,)
x_train, x_test,y_train,y_test = train_test_split(
    x,y , train_size=0.8,random_state=123, shuffle=True,
)
# 2. 모델 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123)
# 3. 훈련 
model.fit(x_train,y_train)
# 4. 평가, 예측
results = model.score(x_test,y_test)
print("처음 결과 : ", results)
print(len(x[0]))
for i in range(x.shape[1]):
    #print('i 값 : ',i)
    # 1. 데이터 
    datasets = load_diabetes()
    x = datasets['data']
    y = datasets['target']
    pca = PCA(n_components=(i+1))
    x = pca.fit_transform(x)
    x_train, x_test,y_train,y_test = train_test_split(
        x,y , train_size=0.8,random_state=123, shuffle=True,
    )
    # 2. 모델 
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=123)
    # 3. 훈련 
    model.fit(x_train,y_train)
    # 4. 평가, 예측
    results = model.score(x_test,y_test)
    print(i+1,"차원 결과 : ", results)
    
'''
처음 결과 :  0.8121690217687418
8
1 차원 결과 :  -0.4412309439201201
2 차원 결과 :  0.046317003872303086
3 차원 결과 :  0.0789494633195541
4 차원 결과 :  0.3241551445937575
5 차원 결과 :  0.5918722922244304
6 차원 결과 :  0.7018597110810503
7 차원 결과 :  0.7786727671384369
8 차원 결과 :  0.7825857242412009 
'''
