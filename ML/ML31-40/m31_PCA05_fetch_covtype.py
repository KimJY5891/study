### 실습 ### 
#for문 써서 한 번에 돌리기
# 기본 결과 : 
# 차원 1개 축소 :
# 차원 2개 축소 : 
 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터 
datasets = fetch_covtype()
print(datasets.feature_names)

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
    datasets = fetch_covtype()
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

'''
