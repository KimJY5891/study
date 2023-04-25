# Linear Discriminant Analysis
# 상민이가 회귀에서 된다고 했다!!! 
# 성호는 y에 라운드 때렸어

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, fetch_california_housing
from tensorflow.keras.datasets import cifar100

# 1. 데이터
x,y = load_diabetes(return_X_y=True)
print(np.unique(y))
# 디아벳은 회귀 값이지만  모두 정수값이라서 클래스로 판단한 것이다.
x,y = fetch_california_housing(return_X_y=True)
# y= np.round(y) # 라운드 처리 하면 정수값으로 변하면서 회귀값을 클래스로 판단하게 되면서  lda가 실행이 된 것이다.
# 라운드 개념이 전처리가 될 수도 있지만, 데이터 조작이 될 수도 있다.
# 의외로 소수점 이하의 값이 중요할 수도 있기 때문에 조심해야한다. 
# ValueError: Unknown label type: (array([4.526, 3.585, 3.521, ..., 0.923, 0.847, 0.894]),)
# 라벨 타입은 잘 모르겠다. 
# 디아벳은 잘 돌아갓는데, 
# lda 알고리즘이 클래스 -1 이다.
print(np.unique(y))
lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis(n_components=3)

x_lda = lda.fit_transform(x,y)
print(x.shape) # (442, 10)

# 결론 : 안된다. 
