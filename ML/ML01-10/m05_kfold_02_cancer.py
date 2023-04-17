import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
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
