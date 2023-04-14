import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터 
datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x,y= load_iris(return_X_y=True)
# 둘 다 가능 
print(x.shape,y.shape)

# 2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree  import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# svm(서포트 벡터 머신)
# model = Sequential()
# model.add(Dense(10, activation='relu',input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3,activation='softmax'))
# model = LinearSVC(C=0.3)
# model = LogisticRegression # 근데 분류임 - 중요 면접에 나옴 
model = DecisionTreeClassifier() # 분류
model = DecisionTreeRegressor() # 회귀 
# 분류 모델이 회귀에 돌아간다. 성능에 문제 있을 수도 있으니 주의
# 옛날 구현모델은 디폴트만해도 잘 먹힌다. 
# 나중에는 뭔가해줘야할듯
# 지금하시는 머신러닝은 단층으로 이루어져있다. 
# 파라미터 c가 클수록 직선, 작으면 더 정교하게 데이터의 영역이 나뉜다. 
# 결국 머신러닝도 선 긋는다. 
# 트리구조 모델들은 앙상블 에서 이상치나 결측치에서 자유롭다.
# 랜덤포레스트, 그 상위 xgbooster
model = RandomForestRegressor()
# 랜덤포레스트 부터슨 튜닝 해주는게 좋다. 
# 하지만 의외로 디폴트가 좋기도한다. 


# 3. 컴파일, 훈련 
# 원핫 귀찮을 땐 spars_categorical_crossentropy로 하면 된다 
# 주의 사항은 0이 있어야하고 아니면 틀어지기 때문에 확인 한 번 해야 한다. 

# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer='adam',
#     metrics=['acc']
#     )
# model.fit(x,y,epochs=100,validation_split=0.2)
model.fit(x,y) # 핏에 컴파일 포함



# 4. 평가 예측
# results = model.evaluate(x,y)
# print(results) # [0.129126638174057, 0.9466666579246521]
results = model.score(x,y)
print(results) # 0.9666666666666667

# 머신러닝은 딥러닝이다.()
# 딥러닝은 머신러닝이다.(x)
# 딥러닝은 머신러닝에 포함 되어잇다.
# 모델안에 알고리즘 되어있다. 
# 문법의 방식은 딥러닝이랑 같다. 
# 케라스가 사이킷런의 문법과 유사하게 만들었다. 
