
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler

# 1.데이터 
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train,y_test = train_test_split(
    x,y, random_state=337,train_size=0.8,stratify=y
)

x_train= RobustScaler().fit_transform(x_train)
x_test= RobustScaler().fit_transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split)
parameter =  {
    "n_estimators" : 100, # 디폴트 100 / 1 ~ inf / 정수
    "learning_rate" : 0.3, # 디폴트 0.3 / 0 ~ 1 / eta
    "max_depth" : 2, # 디폴트 6 / 0 ~ inf / 정수
    "gamma" :0, # 디폴트 0 / 0 ~ inf 
    "min_child_weight" : 1, # 디폴트 1 / 0 ~ inf 
    "subsample" : 0.3, # 디폴트 1 / 0 ~ 1 
    "colsample_bytree" : 1, # 디폴트 / 0 ~ 1 
    "colsample_bylevel": 1 , # 디폴트 / 0 ~ 1 
    "colsample_bynode":1, # 디폴트 / 0 ~ 1 
    "reg_alpha":0, # 디폴트 0 / 0 ~ inf / L1 절대값 가중치 규제 / alpha
    "reg_lambda":1, # 디폴트 1 / 0 ~ inf / L2 제곱 가중치 규제 / lambda
    "random_state" : 337
}

#2. 모델 
model = XGBClassifier(**parameter)

# 3. 훈련
hist = model.fit(
    x_train,y_train,
    eval_set =[(x_train,y_train),(x_test,y_test)], 
    early_stopping_rounds = 10, 
    # eval_metric='logloss', # 이진분류
    # eval_metric='error',  # 이진분류
    # eval_metric='auc',    # 이진분류
    # eval_metric='merror', # 다중 분류 | mlogloss , m : multi
    # 이진분류는 다중 분류에 포함된다. 
    # eval_metric='rmse','mae', 'rmsle' # 회귀  # 되긴 하는데, 회귀에서 가능 
    # 분류 데이터는 회귀 평가지표로 가능하긴하다. 로스이기 기반이기 때문이다.
    # 회귀는 분류 평가 지표로 불가능하다. 
    # xgboost.core.XGBoostError: [11:45:02] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-07593ffd91cd9da33-1\xgboost\xgboost-ci-windows\src\metric\multiclass_metric.cu:34: Check failed: label_error >= 0 && label_error < static_cast<int32_t>(n_class): MultiClassEvaluation: label must be in [0, num_class), num_class=1 but found 1 in label
    # 다중분류에서 사용하라는 놈이라서 에러 남 
    verbose= True 
)

# 4. 평가, 예측 
# print("최상의 매개변수 : ",model.best_params_)
# print("최상의 매개변수 : ",model.best_score_)
result = model.score(x_test,y_test)
print("최종점수 : ", result)
'''
hist = model.fit(
    x_train,y_train,
    eval_set =[(x_train,y_train),(x_test,y_test)], 
    early_stopping_rounds = 10, 
    verbose= True 
) 일 경우 파라미터를 반환할 뿐이다. 
hist :  XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=0, gpu_id=None,
              grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.3, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=2, max_leaves=None,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=337, ... )
'''

print('===========================================================')
hist = model.evals_result()
print('hist : ',hist )
'''
hist :  {'validation_0': - (x_train,y_train)의 발리데이션
OrderedDict(
    [
        ('logloss', [0.4885174797131465, 0.3631903565846957, 0.2823503744798702, 0.22677369163586544, 0.18645844611820284, 0.1557564703972785, 0.13506504045082973, 0.11973436355263323, 0.10867763038847472, 0.09640798955173283, 0.08726670225816113, 0.07883235473022028, 0.0725220569276384, 0.06732338116616829, 0.0624965187458956, 0.06104012734185044, 0.05916541146298687, 0.0551919378754734, 0.05255232055896668, 0.04974933783728425, 0.04745355343655939, 0.04591174384599531, 0.04533781915324853, 0.04369138928455209, 0.04215447004925418, 0.04131990295622477, 0.04052779414552868, 0.04023684239229904, 0.03985124028231539, 0.03918949423612423, 0.03761629770857865, 0.03632570087243189])]), 'validation_1': OrderedDict([('logloss', [0.5167681713376129, 0.42465438785260184, 0.3530213551824553, 0.32152177341151655, 0.3045710120023343, 0.27902398722475036, 0.2610697533216393, 0.25210755011230185, 0.25500128496634333, 0.24088562757038234, 0.24188564673654342, 0.23681165884951488, 0.22554419355532318, 0.22622532473076462, 0.2264955490541628, 0.23054404354964694, 0.23349151539763338, 0.22347608341410624, 0.22490549198221088, 0.22903346566671276, 0.2295727220194398, 
0.22163281241011104, 0.22824391592886267, 0.2300819615596546, 0.2300954141844015, 0.23702203249711984, 0.24927108685829139, 0.25369462032655354, 0.25717033346473617, 0.2585909853782811, 0.26441673518019615, 0.2605844309064355])])}
컴파일에 평가 지표가 있는건데,
컴파일을 안했는데, 평가 지표가 나옴 그러면 디폴트는 로그로스이다.
분류에서는 로그로스
회귀에서는 
'''
'''
subplot을 이용하여 그래프 그림 두 개로 나눠서 선을 그리기
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.title('validation_')
plt.plot(hist['validation_0']['logloss'],  label='Training loss')
plt.subplot(1,2,2)
plt.title('validation_1')
plt.plot(hist['validation_1']['logloss'],  label='Validation loss')
plt.show()
'''
# subplot을 이용하여 그래프 그림 한 개에 선 두 개 넣기 
import matplotlib.pyplot as plt
plt.title('validation')
plt.plot(hist['validation_0']['logloss'],  label='Training loss')
plt.plot(hist['validation_1']['logloss'],  label='Validation loss')
plt.show()
