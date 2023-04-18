# 나누긴했는데 데이터가 한 쪽(0이나 1)으로 쏠려있다면 훈련할때 2값은 잘 안나옴 
# y의 클래스(0,1,2)의 비율로 나뉘는게 좋다. 
# StratifiedGroupKFold - y의 클래스 비율대로 나눠준다. 
# 그리드 서치 - 모든경우의 수를 다 자동적으로 집어넣겟다.  - 케라스2, 텐서플로우에 적용한다. 
# 단점 : 시간이 느리다.
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score,r2_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# 1. 데이터 
x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=337,test_size=0.2
)
n_splits = 5
kfold = StratifiedKFold(n_splits = n_splits,
        # StratifiedKFold - 분류에서만 사용하게 되어있다. 
      shuffle=True,  
      random_state=337,
)

# 2. 모델
model= SVC()

# 3, 4. 컴파일, 훈련, 평가, 예측      
score = cross_val_score(model, x_train,y_train,cv=kfold)
print('r2_score : ',score,'\n 교차검증평균점수 : ',round(np.mean(score),4))
y_pred = cross_val_predict(model,x_test,y_test,cv=kfold)
print('cross_val_predict acc : ',accuracy_score(y_test,y_pred))
#엑큐러시는 편향된 데이터에서 평가데이터로 사용하기 어렵다.
# f1스코어는 편향된 데이터 가능 
