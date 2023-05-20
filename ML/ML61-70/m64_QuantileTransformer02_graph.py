
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer, Normalizer
import matplotlib.pyplot as plt


x,y=make_blobs(
    n_samples=50, # 데이터 50 개로 줄어든다. 
    centers=2, # 중심점을 잡는다. 중심점 24대, 클러스터 갯수  = y의 라벨  
    cluster_std=1, # 클러스터의 표준편차 1 
    random_state=337,
)
# 이 데이터는 랜덤하게 만들지만 완전랜덤은 아니다.
# 가우시안 정규분포를 기준으로 샘플을 만든 예제 . 
# cluster :  두개의 군집을 만들겠다. 

print(x)
print(y)
print(x.shape,y.shape) # (50, 2) (50,)

plt.rcParams['font.family'] = 'Malgun Gothic' #한글  

fig,ax = plt.subplots(3,3,figsize=(10,5))
ax[0,0].scatter(x[:,0], # 모든 행의 0번째 열
            x[:,1],
            c=y,
            edgecolors='black',# 가장자리에 검정색을 넣어라 대충 점에  테두리 넣기
            ) # scatter : 점을 뿌리다. 
ax[0,0].set_title('오리지널')

scaler = StandardScaler() # 오십개라 50개 잡아주는게 좋다. 
# 50의 데이터 넘어도 qt사용가능하지만 
x_train = scaler.fit_transform(x) 
ax[1,0].scatter(x_train[:,0],
            x_train[:,1],
 c=y,edgecolors='black',) 
ax[1,0].set_title(type(scaler).__name__)


scaler = QuantileTransformer(n_quantiles=50) 
x_train = scaler.fit_transform(x) 
ax[1,1].scatter(x_train[:,0], 
            x_train[:,1],
 c=y,edgecolors='black',)
ax[1,1].set_title(type(scaler).__name__)

scaler = MaxAbsScaler() 
x_train = scaler.fit_transform(x) 
ax[2,0].scatter(x_train[:,0], 
            x_train[:,1],
 c=y,edgecolors='black',)
ax[2,0].set_title(type(scaler).__name__)

scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x) 
ax[2,1].scatter(x_train[:,0], 
            x_train[:,1],
 c=y,edgecolors='black',)
ax[2,1].set_title(type(scaler).__name__)

scaler = PowerTransformer() 
x_train = scaler.fit_transform(x) 
ax[2,2].scatter(x_train[:,0], 
            x_train[:,1],
 c=y,edgecolors='black',)
ax[2,2].set_title(type(scaler).__name__)

plt.show() 
