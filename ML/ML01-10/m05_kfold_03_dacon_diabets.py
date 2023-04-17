import numpy as np
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score

# 1. 데이터
path = "./_data/dacon_diabets/"
path_save = "./_save/dacon_diabets/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################                                                                                 #
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)
####################################### 결측치 처리 #######################################                                                                                 #
x = train_csv.drop(['Outcome'],axis=1)#
y = train_csv['Outcome']
print("x.shape : ",x.shape)#(652, 8)
print("y.shape : ",y.shape)#(652, )
n_splits = 5
kfold = KFold(n_splits = n_splits,
      shuffle=True,  
      random_state=123,
      )

# 2. 모델
from sklearn.svm import LinearSVC
model = LinearSVC()

#3, 4. 컴파일, 훈련 ,평가, 예측
scores = cross_val_score(model,x,y,cv =kfold)
print('acc : ',scores,'\n cross_val_score 평균 : ',round(np.mean(scores),4))
