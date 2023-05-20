# 회귀로 만들기 
# 회귀데이터 올인 포문 
# Salcer 6개 올인
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer, Normalizer
# Quantile : 분위수
# 4분위수 이런것과 같은 의미
# QuantileTransformer : 정규분포로 만든다음에 분위수로 나눈다.
# 스탠다드 스케일러 + 민맥스 스케일러 0~1사이
# 좋을 수도 있고 안좋을수도 있다. 
# 가장 끝에 있는 놈들 아웃라이어에 관련 있다. 
# 분위수이기 때문에 아웃라이어에 강하다.
# 될수도 있고 아닐 수도 있다.  
# xgboost가 이상치 자유롭다 근데 이상치 처리해주는게 더 성능이 좋을 확률이 높다. 

import numpy as np
import pandas as pd 
# from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
scaler_name= [
    StandardScaler(),
    MaxAbsScaler(),
    MinMaxScaler(),
    RobustScaler(),
    QuantileTransformer(),
    QuantileTransformer(n_quantiles=1000), #디폴트 1000, 기본 1000분위수임 조절해야한ㅏ.
    PowerTransformer(),
    # PowerTransformer(method='box-cox'),
    #ValueError: The Box-Cox transformation can only be applied to strictly positive data
    # PowerTransformer(method='yeo-johnson'),
    Normalizer()
              ]
'''
    Box-Cox 변환은 비정규 데이터를 정규 데이터로
    변환하는 방법입니다. 엄격하게 긍정적인 데이터에만
    적용할 수 있습니다. 따라서 "Box-Cox 변환은 
    엄격하게 양의 데이터에만 적용할 수 있습니다"라는 
    ValueError가 표시되는 경우 변환하려는 데이터에 
    음수 또는 0 값이 포함되어 있으며 다른 변환 방법을 적용하거나
    Box-Cox 변환에 적합하도록 데이터를 전처리합니다.
'''
# 1. 데이터

x,y = load_iris(return_X_y=True)
for random in range(10000) :
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,random_state=random,train_size=0.8,shuffle=True,
        stratify=y
    )

    for i,v in enumerate(scaler_name) : 

        scaler = v
        x_train =scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        # 2. 모델
        model =RandomForestClassifier()
    
        # 3. 훈련
        model.fit(x_train,y_train) 

        # 4. 평가, 예측 
        result = model.score(x_test,y_test)
        print(f'{random}의 {v} 스케일러의 최종점수 : ',round(result,))
