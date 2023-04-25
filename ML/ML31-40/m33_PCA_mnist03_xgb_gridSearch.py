'''
처음에 성능이 어떻게 좋은지 알 수가 없었다. 
압축하고 나서 압축 결과, 압축 전 결과가 얼마나 매치가 되는지 
5개 차원으로 압축햇는데 원본과 99퍼 일치하면 5개 사용하는게 나을 듯 
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



# 1. 데이터 
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']
pca = PCA(n_components=30)
x = pca.fit_transform(x)
print(x.shape)
# pca는 차원증폭이 되지 읺는다.
# 같은 차원으로 했는데 왜 할까
# 아래와 같은 것잇다. 
pca_EVR = pca.explained_variance_ratio_
# explained_variance_ratio_ = 설명가능한 변화율
print(pca_EVR)
# 30번 돌렸을 때 성능차
print(sum(pca_EVR)) # 0.9999999999999998
# 리스트 형태 인걸 넣으면 리스트 안에 있는 숫자를 다 넣어준다.
print(np.cumsum(pca_EVR))
# sum 더하기
# cumsum : 누적합

'''
[ 
0.98204467 (pca를 한 번 했을 때 원본과 일치율)
0.99822116 (pca를 두 번 했을 때 원본과 일치율)
0.99977867 0.9998996  0.99998788 0.99999453
0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
0.99999999 0.99999999 
1.(15개쯤 햇을 때 데이터의 손실이 없을 것)
1.         1.         1.
1.         1.         1.         1.         1.         1.
1.         1.         1.         1.         1.         1.        ]
이것도 100프로 신뢰하면 안된다.
'''

pca_cumsum = np.cumsum(pca_EVR)
print(pca_cumsum)
import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()
