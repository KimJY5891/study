# 다중공선성
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
# 다중공선성은 통상 스탠다드 스케일러
# 목표 컬럼 간의 상관관계를 보는 것

# 1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

df.info()
print(df.describe())
# 개발자는 10이하, 학자들은 5이하만 괜찮다고 한다. 

y = df['target']
x = df.drop(['target'],axis=1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

vif = pd.DataFrame()
vif['variables'] = x.columns
vif['vif'] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
print(vif)
# vif : 다중공선성을 측정하기 위한 통계적 지표
# 다중공선성 측정할 때 해야 하는 거 
# 1) 스케일링 한다.
# 2) y 넣지 않는다. 

x = x.drop(['Latitude','Longitude'],axis=1)
x_train , x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.8, random_state=337
)

scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.fit_transform(x_test)
# 스케일 적용 안 된 데이터를 사용해야 한다. vif는 그냥 확인용

# 2. 모델 
model = RandomForestRegressor(random_state=337)

# 3. 컴파일, 훈련 
model.fit(x_train,y_train)

# 4. 평가, 예측 
score = model.score(x_test,y_test)
print('r2 : ',r2_score(y_test,model.predict(x_test)))

# 드롭 결과 적기 
