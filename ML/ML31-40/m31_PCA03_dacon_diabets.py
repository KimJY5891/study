### 실습 ### 
#for문 써서 한 번에 돌리기
# 기본 결과 : 
# 차원 1개 축소 :
# 차원 2개 축소 : 
 
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터 
path = "./_data/dacon_diabets/"
path_save = "./_save/dacon_diabets/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################
train_csv = train_csv.dropna()
####################################### 결측치 처리 #######################################                          
x = train_csv.drop(['Outcome'],axis=1)#
y = train_csv['Outcome']
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    #stratify=y
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
    # 1. 데이터 
    train_csv=pd.read_csv(path + 'train.csv',index_col=0)
    test_csv=pd.read_csv(path + 'test.csv',index_col=0)
    ####################################### 결측치 처리 #######################################
    train_csv = train_csv.dropna()
    ####################################### 결측치 처리 #######################################                          
    x = train_csv.drop(['Outcome'],axis=1)#
    pca = PCA(n_components=(i+1))
    x = pca.fit_transform(x)
    y = train_csv['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, shuffle=True, random_state=337, test_size=0.2, 
        #stratify=y
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
처음 결과 :  0.24566859979101374
1 차원 결과 :  -0.004042450365726147
2 차원 결과 :  0.06611167711598764
3 차원 결과 :  0.07010190700104513
4 차원 결과 :  0.12487003657262297
5 차원 결과 :  0.1958352142110763
6 차원 결과 :  0.2143456374085685
7 차원 결과 :  0.2581594566353187
8 차원 결과 :  0.28095783699059573
'''
