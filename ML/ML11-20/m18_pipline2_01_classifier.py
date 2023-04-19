# 삼중 포문!! 
# 1. 데이터 
# 2. 스케일
# 3. 모델 

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_wine, load_digits, load_breast_cancer, fetch_covtype, load_iris
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# 삼중 포문!! 
# 1. 데이터 
# 2. 스케일
# 3. 모델 

from sklearn.datasets import  load_boston,fetch_california_housing
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

dataset_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
    load_digits(return_X_y=True),
]
data_list_name = ['캘리포니아',
                  '캔서',
                  '와인',
                  '디지트']
scaler_list = [MinMaxScaler(),StandardScaler(),MaxAbsScaler(),RobustScaler() ]
for i,v in enumerate(dataset_list):
    # 1. 데이터
    x,y = v
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, train_size=0.8, shuffle=True, random_state=337
    )
    # 2. 모델 구성
    for j in scaler_list :
        try:
            max_score = 0
            allAlgoritms = all_estimators(type_filter='classifier')
            for (name, algorithms) in allAlgoritms:
                model = algorithms()
                model = make_pipeline(j,algorithms())
                # 3. 훈련
                model.fit(x_train,y_train)
                #4. 평가, 예측 
                results = model.score(x_test,y_test)
                y_pred = model.predict(x_test)
                acc = accuracy_score(y_test,y_pred)
                print("==================================================")
                print("데이터 : ", data_list_name[i])
                print("acc : ",acc)
                print("모델 : ", name)
                print("점수 : ", results)
                print("스케일러 : ", str(j))
                if max_score < results :
                    max_score = results
                    max_model = name
                    max_data= data_list_name[i]
                    max_scaler = j
        except:
            continue
