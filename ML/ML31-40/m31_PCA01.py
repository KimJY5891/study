# 차원 축소 ( 컬럼 축소 )
# 삭제하는 개념 아니고 압축하는 개념 
# 열 개 컬럼 중에서 하나가 쓰레기여도 압축하는거임
# 특성을 압축하다보니까 원본보다 성능이 떨어져야하는데, 
# 하지만 압축을 시켰을 때, 
# 0이 5개라서 압축할 때 1개로 만듦 
# 예를 들어서 MNIST가 0이 많을 때,  압축하면 오히려 성능이 좋아질 수 가 있다. 
# PCA로  압축했을 때 성능이 좋아지는경우도 있다. 
# X만압축한다. 
# 타겟값은 통상 Y 
# 10개의 컬럼을 2개로 압축햇다. 
# 압축보다는 차원축소가 더 맞는 말 
# 차원축소를 할 때, 10개의 컬럼에서 2개로 압축했으면 
# 차원축소할 때 Y를 사용하지 않았고, 
# 타켓값을 없이 차원축소를 했기 때문에 비지도학습
# 2개로 축소한 걸로 Y값을 찾는거면 스케일링한 걸로 볼 수도 있다.
'''
pca 개념 
여러 데이터가 점으로 있을 대 선을 그어서 여러 데이터와 선위에 자리로
맵핑하는 것 
선쪽으로 데이터 움직이기 
임베딩과 비슷 
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터 

datasets = load_diabetes()
datasets = load_breast_cancer()
print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
print(x.shape,y.shape) #(442, 10) (442,)

pca = PCA(n_components=7)
x = pca.fit_transform(x)
# print(x.shape) # (442, 5)
# 차원 줄었다.
x_train, x_test,y_train,y_test = train_test_split(
    x,y , train_size=0.8,random_state=123, shuffle=True,
) # 디폴트 있음 

# 2. 모델 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123)

# 3. 훈련 
model.fit(x_train,y_train)

# 4. 평가, 예측
results = model.score(x_test,y_test)
print("결과 : ", results)

# none 결과 :  0.5260875642282989
# 
