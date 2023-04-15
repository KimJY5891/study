#분류 만들어

import numpy as np 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
import sklearn as sk


# 1. 데이터
x,y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, test_size=0.2
)
338
scaler=RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델 구성

model = RandomForestClassifier(n_jobs=4)
# allAlgoritms = all_estimators(type_filter='regressor')
allAlgoritms = all_estimators(type_filter='classifier')
print('allAlgoritms : ',allAlgoritms)

print(sk.__version__) # 1.0.2
print(len(allAlgoritms)) #55

for (name, algoritm) in allAlgoritms :
    try: # 에외처리 
        model = algoritm()
        model.fit(x_train,y_train)
        # 3. 훈련
        model.fit(x_train,y_train) # 모델마다 형식이 다르다.
        # 4. 평가, 예측
        results = model.score(x_test,y_test)
        print( name ,'의 정답률 : ',results)
        
        if max_r2 < results:
            max_r2 = results
            max_name = name
        # y_pred = model.predict(x_test)
        # r2_score=r2_score(y_test,y_pred)
        # print("r2_score : ",r2_score)    
    except :  # 에러가 있을 때 실행하고 나서 다시 for문 실행
        print(name,' : 에러 모델')#에러가 난 모델은 안에 파라미터 넣어줘야한다. 
print('===================================')
print('최고 모델 : ',max_name, max_r2)
print('===================================')
