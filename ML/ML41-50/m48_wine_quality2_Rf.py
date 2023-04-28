# 와인의 컬리티를 맞추기 
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score,mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler #전처리
import pprint
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
def outliers(data_out) :  # 이상치를 찾아주는 함수
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 :" , quartile_1) 
    print("q2 :" , q2)             
    print("3사분위 :" , quartile_3) 
    iqr = quartile_3 - quartile_1 
    print("iqr : ",iqr)
    lower_bound = quartile_1 - (iqr * 1.5) 
    upper_bound = quartile_3 + (iqr * 1.5) 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

col_name = ['quality', 'fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
       'type']
'''
# 1. 데이터
1) 
2) 이상치 nan화
3) 결측치 제거 
4) 열 제거 
# 
'''

# y값 퀄리티 
# 이것의 문제 : 
# 비율이 나쁘다. 
# 대부분의 연산이 정보가 많은 클라스로 정답이 몰리게 된다. 
# 증폭을 하면 9등급이 400배증폭된다. 허수가 너무 많음 
# 방법 중 하나로 9등급
# 와인의 질을 맞추는 문제
# 6 2416
# 5 1788
# 7 924
# 4 186
# 8 152
# 3 26
# 9 5

#1. 데이터

path = "./_data/dacon_wine/"
path_save = "./_save/dacon_wine/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print("train_csv",train_csv)
print("train_csv",train_csv.shape) #(5497, 13)
print("train_csv",train_csv.columns) #(5497, 13)
test_csv= pd.read_csv(path+ 'test.csv', index_col = 0)
print(test_csv)
print(test_csv.shape) #(1000, 12)

#라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
aaa = le.fit_transform(train_csv['type'])
print(aaa.shape) # (5497,)
print(type(aaa)) # <class 'numpy.ndarray'>
# print(np.unique(aaa,return_count=True)) # 
train_csv['type'] = aaa #다시 집어넣기
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

####################################### 이상치 처리 #######################################
train_csv = train_csv.dropna()
####################################### 이상치 처리 #######################################
####################################### 결측치 처리 #######################################

####################################### 결측치 처리 #######################################

x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']
print('y의 라벨 값 :',np.unique(y)) #[3 4 5 6 7 8 9]
print("x.shape : ",x.shape)#(5497, 12)
print("y.shape : ",y.shape)#(5497,)
y = to_categorical(y)
print(y)
y=np.where(y>=7,y-3,y-4)
y= y-3
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.85,shuffle=True,random_state=1234
)

# scaler = StandardScaler() # 최적의 n_estimators : 600, 
# scaler = MinMaxScaler() #  {'n_estimators': 600
# scaler = MaxAbsScaler() # {'n_estimators': 600}
# scaler = RobustScaler() # 
# non-scaler일 경우, {'n_estimators': 600}
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델 구성
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

model =RandomForestClassifier()

# 3. 컴파일 훈련 
model.fit(x_train,y_train)

# 4. 평가 예측
# print("최적의 매개 변수 : ", model.best_estimator_)
# print("최적의 파라미터 : ", model.best_params_)
# print("best_score : ",model.best_score_)
print("model.score : ",model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('acc_score : ',accuracy_score(y_test,y_pred))
# y_pred_best =model.best_estimator_.predict(x_test)
# print('최적의 튠 acc : ',accuracy_score(y_test,y_pred_best))
# model.score :  0.5842424242424242
# acc_score :  0.5842424242424242
# 5. 제출
# y_submit = np.round(model.predict(test_csv))
# submissions = pd.read_csv(path+'sample_submission.csv',index_col=0)
# submissions['quality'] = y_submit
# submissions.to_csv(path_save+'submit_0428.csv')


