import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터 

path = "./_data/dacon_diabets/"
path_save = "./_save/dacon_diabets/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################   
train_csv = train_csv.dropna()
####################################### 결측치 처리 #######################################                   
x = train_csv.drop(['Outcome'],axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337,train_size=0.8,shuffle=True,
    stratify=y
)

scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model = BaggingClassifier(
#  BaggingClassifier(
    RandomForestClassifier(),
    n_estimators=20,
    n_jobs=-1,
    random_state=337,
    # bootstrap=False
    bootstrap=True
# )
)
# model= RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측 
y_pred = model.predict(x_test)
print('model.score : ',model.score(x_test,y_test))
print('acc : ',accuracy_score(y_test,y_pred))
'''
랜덤  포레스트 
model.score :  0.7862595419847328
acc :  0.7862595419847328
'''
'''
배깅 -랜덤 포레스트 - bootstrap = True
model.score :  0.7709923664122137
acc :  0.7709923664122137
'''
'''
배깅 -랜덤 포레스트 - bootstrap = False
model.score :  0.7786259541984732
acc :  0.7786259541984732
'''
'''
배깅 - 배깅 -랜덤 포레스트 - bootstrap = True
model.score :  0.7862595419847328
acc :  0.7862595419847328
'''
