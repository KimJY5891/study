# 그리드 = 그물망
# 그물망처럼 찾겠다.
# 파라미터 전체를 찾겠다 다 하겠다.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score # 평가지표 
from sklearn.svm import SVC


# 1. 데이터 
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, 
    stratify=y# 0,1,2가 골고루 분배된다. 
)
gamma = [0.001,0.01,0.1,1,10,100]
C = [0.001,0.01,0.1,1,10,100]
# c를 하게되면 곡선

# 2. 모델 
# 우리는 한개의 값만 넣어야한다. 그래서 for 문을 돌려야한다.
max_score = 0 # max란 값이 파이썬에 예약어가 있다. 
for i in gamma :
    for j in C : 
        model = SVC(gamma=i,C=j)

        # 3. 컴파일 훈련 
        model.fit(x_train,y_train,)

        # 4. 평가 예측
        # model.score
        # 분류일때는 accuracy_scor
        # 회귀일 대는 r2 스코어
        score = model.score(x_test,y_test)
        print("acc : ",score) # acc :  0.9666666666666667
        
        if max_score < score : 
            max_score = score
            score = model.score(x_test,y_test)
            # 기존값과 최근 값중에서 가장 높은거 비교해서 높은 애 저장 
            # 최적의 파라미터도 알아야한다. 
            best_paramiters = {'gamma' : i , 'C' :j }
            # 존재하는 값을 딕셔너리 형태로 저장한다. 
            # 스코어가 갱신되지 않으면 더 돌지 않는다. 
print("최고점수 : ",max_score)
print("최적의 매개변수(=파라미터) : ", best_paramiters)


