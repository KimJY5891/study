from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor
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
train_csv = train_csv.dropna()
# csv_all = pd.concat([train_csv,test_csv],axis=0)
# le = LabelEncoder()
# csv_all['SMILES'] = le.fit_transform(csv_all['SMILES']) # 0과 1로 변화
# train_csv['SMILES'] = le.transform(train_csv['SMILES']) # 0과 1로 변화
# test_csv['SMILES'] = le.transform(test_csv['SMILES']) # 0과 1로 변화

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

# 최적의 k 값을 찾기 위해 중첩 k-겹 교차 검증을 수행하는 함수
def find_best_k(x_train, y_train, search_space, max_k=10):
    best_k = None
    best_rmse = float('inf')
    
    for k in range(2, max_k + 1):
        kfold = KFold(
            n_splits=k,
            shuffle=True,
            random_state=128,
        )
                
        # 하이퍼파라미터 최적화 실행
        trials_val = Trials()
        best_params = fmin(
            fn=lambda params: RF_hamsu(params, kfold),
            space=search_space,
            algo=tpe.suggest,
            max_evals=128,
            trials=trials_val,
            rstate=np.random.default_rng(seed=10),
        )
        
        # 교차 검증을 사용하여 성능 평가
        model = RandomForestRegressor(**best_params)
        rmse_scores = -cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
        avg_rmse = np.mean(rmse_scores)
        
        print(f"k={k}일 때 평균 RMSE: {avg_rmse}")
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_k = k
    
    return best_k, best_rmse

search_space = {
    'n_estimators': hp.quniform('n_estimators', 10, 2000, 1),
    'criterion': hp.choice('criterion', ['absolute_error', 'poisson', 'friedman_mse', 'squared_error']),
    # 'criterion': hp.choice('criterion', ['mse', 'mae']),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
    'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', [None, 10, 20, 30, 40, 50]),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 0.2),
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'oob_score': hp.choice('oob_score', [True, False]),
    'n_jobs': hp.choice('n_jobs', [None, 1, 2, 3, 4]),
    'random_state': hp.choice('random_state', [None, 42, 56, 71]),
    'verbose': hp.quniform('verbose', 0, 5, 1),
    'warm_start': hp.choice('warm_start', [True, False]),
    'ccp_alpha': hp.uniform('ccp_alpha', 0.0, 0.1)
}



# RF_hamsu 함수를 수정하여 kfold를 인자로 받을 수 있게 함
def RF_hamsu(search_space, kfold):
    bootstrap = search_space['bootstrap']
    oob_score = search_space['oob_score']

    if oob_score and not bootstrap:
        return {'status': 'fail', 'reason': 'oob_score=True requires bootstrap=True'}

    params = {
        'n_estimators': int(round(search_space['n_estimators'])),
        'criterion': search_space['criterion'],
        'max_depth': int(round(search_space['max_depth'])),
        'min_samples_split': search_space['min_samples_split'],
        'min_samples_leaf': search_space['min_samples_leaf'],
        'min_weight_fraction_leaf': search_space['min_weight_fraction_leaf'],
        'max_features': search_space['max_features'],
        'max_leaf_nodes': search_space['max_leaf_nodes'],
        'min_impurity_decrease': search_space['min_impurity_decrease'],
        'bootstrap': bootstrap,
        'oob_score': oob_score,
        'n_jobs': search_space['n_jobs'],
        'random_state': search_space['random_state'],
        'verbose': int(search_space['verbose']),
        'warm_start': search_space['warm_start'],
        'ccp_alpha': search_space['ccp_alpha']
    }
    model = RandomForestRegressor(**params)

    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 평가, 예측
    y_pred = model.predict(x_test)
    result_mse = mean_squared_error(y_test, y_pred)
    return result_mse


best_k, best_rmse = find_best_k(x_train, y_train, search_space)
print(f"최적의 k: {best_k}, 최적의 평균 RMSE: {best_rmse}")
