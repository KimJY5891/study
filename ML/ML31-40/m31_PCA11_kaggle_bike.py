

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터
path = "./_data/kaggle_bike/"
path_save = "./_save/kaggle_bike/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 ####################################### 
train_csv = train_csv.dropna()
####################################### 결측치 처리 ####################################### 
x = train_csv.drop(['count'],axis=1)#
y = train_csv['count']
print("x.shape : ",x.shape)#(652, 8)
print("y.shape : ",y.shape)#(652, )
x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=123,test_size=0.2
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
    # 1. 데이터
    path = "./_data/kaggle_bike/"
    path_save = "./_save/kaggle_bike/"
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
        x,y,random_state=123,test_size=0.2
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
 처음 결과 :  0.9998037292625677
1 차원 결과 :  0.9527647756359248
2 차원 결과 :  0.9996970900540822
3 차원 결과 :  0.9997473158405265
4 차원 결과 :  0.9997412365995952
5 차원 결과 :  0.9997155257589964
6 차원 결과 :  0.9996984411836431
7 차원 결과 :  0.9996917832773888
8 차원 결과 :  0.999677142846171
9 차원 결과 :  0.9996582988554953
10 차원 결과 :  0.9996462876853105   
'''
