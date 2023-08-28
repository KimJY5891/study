import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

# 1. 데이터

path_data = '/content/drive/MyDrive/대회/medical/_data/'
path_save = '/content/drive/MyDrive/대회/medical/_save/'
train_csv = pd.read_csv(path_data+'train.csv',index_col=0)
print(train_csv.shape) #(3498, 10)
print(train_csv.info())
print(train_csv.describe())
test_csv = pd.read_csv(path_data+'test.csv',index_col=0)
test_csv = test_csv.drop(['SMILES'],axis=1)
train_csv = train_csv.dropna()
# csv_all = pd.concat([train_csv,test_csv],axis=0)
# le = LabelEncoder()
# csv_all['SMILES'] = le.fit_transform(csv_all['SMILES']) # 0과 1로 변화
# train_csv['SMILES'] = le.transform(train_csv['SMILES']) # 0과 1로 변화
# test_csv['SMILES'] = le.transform(test_csv['SMILES']) # 0과 1로 변화

y = train_csv['MLM']
x = train_csv.drop(['MLM','HLM','SMILES'],axis=1)
print(y.shape) # (3498,)
print(x.shape) # (3498, 9)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9, random_state=337
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model_mlm = XGBRegressor(colsample_bylevel = 0.618919336792986,
                         colsample_bynode = 0.9266663839365478,
                         colsample_bytree = 0.7090617117264464,
                         gamma = 6, learning_rate = 0.24837155313744627,
                         max_depth =  5, min_child_weight = 124.22668999343279,
                         n_estimators = 506, reg_alpha = 3.7288788573065568,
                         reg_lambda = 33.35574544888175, subsample = 0.7127270507946109
                         )

# 3. 컴파일, 훈련
num_boost_round = int(10)
model_mlm.fit(
    x_train,y_train,
    eval_set=[(x_train, y_train),(x_test, y_test)],
    eval_metric="rmse",
    # early_stopping_rounds=100,
    verbose=10,
    )

# 4. 훈련, 예측
resultl_mlm = model_mlm.score(x_test,y_test)
print(" result: ", resultl_mlm)
y_predl_mlm = model_mlm.predict(x_test)
y_test_mlm = y_test

# 1. 데이터

y = train_csv['HLM']
x = train_csv.drop(['MLM','HLM','SMILES'],axis=1)

print(y.shape) # (3498,)
print(x.shape) # (3498, 9)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9, random_state=337
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. Sequential
model_hlm = XGBRegressor(
    colsample_bylevel = 0.6581616104905023, colsample_bynode = 0.7885503672739463,
    colsample_bytree = 0.6280661478064626, gamma = 9, learning_rate = 0.01603094647748067,
    max_depth = 15, min_child_weight = 37.76883887147575, n_estimators = 471,
    reg_alpha = 9.97589572579176, reg_lambda = 32.48410958809254, subsample = 0.9261100061120672
)

# 3. 컴파일, 훈련
num_boost_round = int(10)
model_hlm.fit(
    x_train,y_train,
    eval_set=[(x_train, y_train),(x_test, y_test)],
    eval_metric="rmse",
    # early_stopping_rounds=100,
    verbose=10,
          )

# 4. 훈련, 예측
result_hlm = model_mlm.score(x_test,y_test)
print(" result: ", result_hlm)
y_pred_hlm = model_mlm.predict(x_test)
y_test_hlm = y_test
loss = 0.5 *rmse(y_test_mlm,y_predl_mlm)+0.5 *rmse(y_test_hlm,y_pred_hlm)
print(" loss: ", loss)

# 5. 제출
submission = pd.read_csv(path_data +'sample_submission.csv',index_col=0)
submission['MLM'] = model_mlm.predict(test_csv)
submission['HLM'] = model_hlm.predict(test_csv)
submission.to_csv(path_save+'0829_none_es.csv')
