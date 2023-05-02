import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso, Ridge
# L1 규제 : 가중치 절대값 규제
# L2 규제 : 가중치 제곱 규제 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

x,y =load_linnerud(return_X_y=True)
# print(x.shape,y.shape) # (20, 3) (20, 3)

model = Ridge()
model.fit(x,y)
y_pred =model.predict(x)
print(model.__class__.__name__,'_mean_absolute_error : ', round(mean_absolute_error(y,y_pred),4))
print('predict : ', model.predict([[2,110,43]])) # 행렬이라서 차원 맞춰주느라고 
# [138.  33.  68.]예상 
# 결과 : [[187.32842123  37.0873515   55.40215097]]
model = XGBRegressor()
model.fit(x,y)
y_pred =model.predict(x)

print(model.__class__.__name__,'_mean_absolute_error : ', round(mean_absolute_error(y,y_pred),4))
 # 0.9999999567184008
print('predict : ', model.predict([[2,110,43]])) # [[138.00215   33.001656  67.99831 ]]

# model = LGBMRegressor()
# model.fit(x,y)
# print('model.score : ', model.score(x,y))
# print('predict : ', model.predict([[2,110,43]]))
#ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# 훈련을 3번해야한다. 
# 각 열 3번 훈련해서 값을 내놓는다. 

model = MultiOutputRegressor(LGBMRegressor())
model.fit(x,y)
y_pred =model.predict(x)
print(model.__class__.__name__,'_mean_absolute_error : ', round(mean_absolute_error(y,y_pred),4))
print('predict : ', model.predict([[2,110,43]]))
# MultiOutputRegressor을 랩핑하는 개념으로 한 번 감싸준다. 
# 

# model = CatBoostRegressor()
# model.fit(x,y)
# print(model.__class__.__name__,'_mean_absolute_error : ', round(mean_absolute_error(y,y_pred),4))
# print('predict : ', model.predict([[2,110,43]]))
# _catboost.CatBoostError: C:/Go_Agent/pipelines/BuildMaster/catboost.git/catboost/private/libs/target/data_providers.cpp:612: Currently only multi-regression, multilabel and survival objectives work with multidimensional target
# 해결방법 
# 1) 3번 훈련 
# 2)MultiOutputRegressor 사용 

model = MultiOutputRegressor(CatBoostRegressor())
model.fit(x,y)
y_pred =model.predict(x)
print(model.__class__.__name__,'_mean_absolute_error : ', round(mean_absolute_error(y,y_pred),4))
print('predict : ', model.predict([[2,110,43]]))

model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x,y)
y_pred =model.predict(x)
print(model.__class__.__name__,'_mean_absolute_error : ', round(mean_absolute_error(y,y_pred),4))
print('predict : ', model.predict([[2,110,43]]))


