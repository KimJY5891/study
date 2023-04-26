import numpy as np
import pandas as pd
import sklearn as sk

print(sk.__version__)
data =pd.DataFrame([[2,np.nan,6,8,10],
                    [2,4,np.nan,8,np.nan],
                    [2,4,6,8,10],
                    [np.nan,4,np.nan,8,np.nan]]
                   ).transpose()

print(data) # (5, 4)
data.columns = ['x1','x2','x3','x4']
print(data) # (4, 5)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# 실험적인 모델 사용할 때 필요한 import 
# impute : 돌리다
# 결측치에 대한 책임을 돌릴 것 같다.
imputer = SimpleImputer() # 디폴트 = 평균
imputer = SimpleImputer(strategy='mean')
imputer = SimpleImputer(strategy='median')
imputer = SimpleImputer(strategy='most_frequent') # 최빈값 다 똑같은 빈도일 경우 가장 작은 값을 넣는다. 
imputer = SimpleImputer(strategy='constant') # 0 # 끊임없는 
imputer = SimpleImputer(strategy='constant',fill_value=7777)
imputer = KNNImputer() # knn알고리즘으로 평균값을 찾아낸 것이다. 
#k:최근접,  n : 네이버 , 최근접 이웃에서 한정하는것  
imputer = IterativeImputer() # 이거는 약해서 위나 아래로만 사용 # 단순 선형회귀 # 하지만 대회나 그런 곳에서 이상치나 결측치 맞출 때 안맞을 수도 있을 수 있다. 
imputer = IterativeImputer(estimator=DecisionTreeRegressor()) 
imputer = IterativeImputer(estimator=XGBRegressor()) 
# 아직 실험적인것 
# 값이 선형 회귀 모델과 비슷하게 나왔다.
print(imputer)

data2 = imputer.fit_transform(data)
print(data2)


