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
datasets = load_breast_cancer()
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
    datasets = load_breast_cancer()
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
처음 결과 :  0.9162159037754761
1 차원 결과 :  0.6187760775141997
2 차원 결과 :  0.8044821917808219
3 차원 결과 :  0.8009856331440026
4 차원 결과 :  0.9056957567657868
5 차원 결과 :  0.9016697627798196
6 차원 결과 :  0.8981198797193451
7 차원 결과 :  0.8899117273638489
8 차원 결과 :  0.8908068159037754
9 차원 결과 :  0.8892984964918142
10 차원 결과 :  0.8914733711994653
11 차원 결과 :  0.8921361176077514
12 차원 결과 :  0.8974609421984631
13 차원 결과 :  0.8968553291012362
14 차원 결과 :  0.8911458068827264
15 차원 결과 :  0.8884795856999665
16 차원 결과 :  0.8908334781156031
17 차원 결과 :  0.8904030738389576
18 차원 결과 :  0.8935949214834614
19 차원 결과 :  0.8889709321750752
20 차원 결과 :  0.8882396257935181
21 차원 결과 :  0.8846821249582358
22 차원 결과 :  0.8791325760106916
23 차원 결과 :  0.882225392582693
24 차원 결과 :  0.8852496491814233
25 차원 결과 :  0.8785917139993318
26 차원 결과 :  0.8769538924156365
27 차원 결과 :  0.884206014032743
28 차원 결과 :  0.8718880721683929
29 차원 결과 :  0.8695684597393919
30 차원 결과 :  0.8699226862679585
'''
