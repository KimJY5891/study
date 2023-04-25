

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터
path = "./_data/dacon_ddarung/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################
train_csv = train_csv.dropna()
####################################### 결측치 처리 #######################################
x = train_csv.drop(['count'],axis=1)#
y = train_csv['count']
x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=123,test_size=0.2,shuffle=True
)
# 2. 모델 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123)
# 3. 훈련 
model.fit(x_train,y_train)
# 4. 평가, 예측
results = model.score(x_test,y_test)
print("처음 결과 : ", results)

for i in range(x.shape[1]):
    #print('i 값 : ',i)
    path = "./_data/dacon_ddarung/"
    train_csv=pd.read_csv(path + 'train.csv',index_col=0)
    test_csv=pd.read_csv(path + 'test.csv',index_col=0)
    ####################################### 결측치 처리 #######################################
    train_csv = train_csv.dropna()
    ####################################### 결측치 처리 #######################################
    x = train_csv.drop(['count'],axis=1)#
    pca = PCA(n_components=(i+1))
    x = pca.fit_transform(x)
    y = train_csv['count']
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,random_state=123,test_size=0.2,shuffle=True
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
처음 결과 :  0.7932342581722173
1 차원 결과 :  -0.26917844142607183
2 차원 결과 :  0.09707754599708784
3 차원 결과 :  0.2353812467908073
4 차원 결과 :  0.29494787387754695
5 차원 결과 :  0.62252151710199
6 차원 결과 :  0.6935666626368776
7 차원 결과 :  0.6754821123352452
8 차원 결과 :  0.6784454950446617
9 차원 결과 :  0.6936276869466653 
'''
