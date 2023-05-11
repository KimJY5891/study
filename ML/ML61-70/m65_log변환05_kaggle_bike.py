from sklearn.datasets import fetch_california_housing, load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score

# 1. 데이터
path = "./_data/kaggle_bike/"
path_save = "./_save/kaggle_bike/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 ####################################### 
train_csv = train_csv.dropna()
####################################### 결측치 처리 ####################################### 
x = train_csv.drop(['count'],axis=1)
y = train_csv['count']
# 그래프 확인
# x.plot.box()
print(train_csv.columns)
'''
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
'''
# train_csv.hist(bins=50)
# plt.show()
# test_csv.hist(bins=50)
# plt.show()
log_list = ['humidity', 'windspeed','casual','registered']
for i,v in enumerate(log_list) :
    x[v] = np.log1p(x[v])
y = np.log1p(y)

x_train , x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.8, random_state=337
)

y_train_log = np.log1p(y_train)
y_test_log =  np.log1p(y_test)

# 2. 모델 
model = RandomForestRegressor(random_state=337)


# 3. 컴파일, 훈련 
model.fit(x_train,y_train_log)

# 4. 평가, 예측 
score = model.score(x_test,y_test)
print('로그 -> 지수r2 : ',r2_score(y_test,np.expm1(model.predict(x_test))))
# 로그 -> 지수r2 :  0.9997210585807821
# 원래 0.9996883439837274

