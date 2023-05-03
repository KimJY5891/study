import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer,fetch_california_housing, load_iris, load_wine,load_digits,fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor, XGBClassifier 
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor, CatBoostClassifier
def model_function(max_depth,num_leaves,min_child_weight,min_child_sample,subsample,colsample_bytree,max_bin,reg_alpha,reg_lambda) : 
    parameter = {
        "max_depth": int(max_depth), 
        "num_leaves" : int(num_leaves),
        "min_child_weight" : int(min_child_weight), 
        "min_child_sample" : int(min_child_sample),
        "subsample" : float(subsample), 
        "colsample_bytree" : float(colsample_bytree), 
        "max_bin" : int(max_bin), 
        "reg_alpha" :float(reg_alpha), 
        "reg_lambda" :float(reg_lambda)
    }
    model= LGBMRegressor(**parameter)         
    # model = LGBMClassifier(max_depth=int(max_depth), 
    #                        num_leaves=int(num_leaves),
    #                        min_child_weight=int(min_child_weight), 
    #                        min_child_sample=int(min_child_sample),
    #                        subsample=subsample, 
    #                        colsample_bytree=colsample_bytree, 
    #                        max_bin=int(max_bin), 
    #                        reg_alpha=reg_alpha, 
    #                        reg_lambda=reg_lambda)
    # 실수만 받는다.
    # 3. 훈련 
    model.fit(
        x_train,y_train
              )
    # 4. 평가, 예측 
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    return r2

# 1. 데이터 
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337, train_size=0.8, stratify=y
)

# scaler = RobustScaler()
# x_train= scaler.fit_transform(x_train)
# x_test= scaler.transform(x_test)

baysian_params = {
    "max_depth" : (3,16), 
    "num_leaves" : (24,64),
    "min_child_weight" : (10,200), 
    "min_child_sample" : (1,50),
    "subsample" : (0.5,1), 
    "colsample_bytree" : (0.5,1), 
    "max_bin" : (10,500), 
    "reg_alpha":(0.001,10), 
    "reg_lambda":(0.01,50)
}

from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f=model_function,
    pbounds=baysian_params,
    random_state=337
)
# from bayes_opt.util import UtilityFunction
# utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
optimizer.maximize(init_points=2,
                    n_iter=10,
                    # acquisition_function=utility
                    )
print(optimizer.max)




