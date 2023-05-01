import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score

# 1. 데이터 

x,y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337,train_size=0.8,shuffle=True,
    # stratify=y
)
scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier

model = BaggingRegressor(
# BaggingRegressor(
    RandomForestRegressor(),
    n_estimators=20,
    n_jobs=-1,
    random_state=337,
    #bootstrap=True
    bootstrap=False
)
# )
# model= RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측 
y_pred = model.predict(x_test)
print('model.score : ',model.score(x_test,y_test))
print('r2_score : ',r2_score(y_test,y_pred))
'''
랜덤  포레스트 
model.score :  0.3980870392573419
r2_score :  0.3980870392573419
'''
'''
배깅 -랜덤 포레스트 - bootstrap = False
model.score :  0.4174246079474776
r2_score :  0.4174246079474776
'''
'''
배깅 -랜덤 포레스트 - bootstrap = True
model.score :  0.4409394877624232
r2_score :  0.4409394877624232
'''
'''
배깅 - 배깅 -랜덤 포레스트 - bootstrap = True
model.score :  0.441170017528996
r2_score :  0.441170017528996
'''
