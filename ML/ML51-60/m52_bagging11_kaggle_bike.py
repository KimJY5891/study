import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score

# 1. 데이터 

path = "./_data/kaggle_bike/"
path_save = "./_save/kaggle_bike/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 ####################################### 
train_csv = train_csv.dropna()
####################################### 결측치 처리 ####################################### 
x = train_csv.drop(['count'],axis=1)#
y = train_csv['count']

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
    # bootstrap=True
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
model.score :  0.9997102487264394
r2_score :  0.9997102487264394
'''
'''
배깅 -랜덤 포레스트 - bootstrap = False
model.score :  0.9997283839439782
r2_score :  0.9997283839439782  
'''
'''
배깅 -랜덤 포레스트 - bootstrap = True
model.score :  0.9996551098440819
r2_score :  0.9996551098440819 
'''
'''
배깅 - 배깅 -랜덤 포레스트 - bootstrap = True
model.score :  0.9994664304593186
r2_score :  0.9994664304593186
'''
