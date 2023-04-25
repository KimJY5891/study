# Linear Discriminant Analysis
# 선형 판별식 분석
# 피시에이는 데이터의 반향성에 다라서 선을 긋고 
# lda는 각 데이터의 클래스별로 매치를 시킨다.
# 실질적으로는 지도 학습 
# y가 필요하다. 
# 데이터의 클래스에 따라 가른다. 
# 차원 축소 
# 회귀에서 아마도 안될 것 같음 찾아봐야함 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits

# 1. 데이터 
x,y = load_iris(return_X_y=True)
x,y = load_breast_cancer(return_X_y=True)
x,y = load_digits(return_X_y=True)
print(x.shape)
# pca =PCA(n_components=3)
# pca 디폴트 값 전체가 그대로 나오는 거 
# x = pca.fit_transform(x)
# print(x.shape) # 

lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis(n_components=3)
# n_components는 클래스의 갯수 빼기 하나 이하로 가능하다! 
x_lda = lda.fit_transform(x,y)
# lda가 더 좋은 경우가 있다.
# (150, 2 )
print(x.shape)
# 값의 종류가가 3개라서 선을 2개 그은 것 
# 클래스 만큼 자른다. 
# 그래서 디폴트가 2로 된거

