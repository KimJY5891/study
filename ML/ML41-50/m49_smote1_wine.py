# 데이터 증폭
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score #
from imblearn.over_sampling import SMOTE
# pip install imbearn
# 클래스가 불균형할 대 증폭시켜준다. 
# 1. 데이터 
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape,y.shape) # (178, 13) (178, 1)
print(np.unique(y,return_counts =True))# 넘파이로 아니면 
print(pd.Series(y).value_counts().sort_index()) # 와이값이 판다스의 
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 몇개 죽이고 내가 원하는거 임의로 살린다. 
'''
x = x[:-25]
y = y[:-25]
print(x.shape,y.shape) # (153, 13) (153,)
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2]
'''
print(np.unique(y,return_counts =True))# 넘파이로 아니면 
print(pd.Series(y).value_counts().sort_index())
'''
0    59
1    71
2    23

'''


x_train, x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.75,shuffle=True,random_state=3377,
    stratify= y
)
print(pd.Series(y_train).value_counts().sort_index())
'''
0    44
1    53
2    17
53개를 기준으로 증폭한다.
증폭
1. copy 하기  
2. 근사치들을 이용해서 비슷한 데이터를 만들고 
'''

#2. ahepf
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# 3. 훈련

model.fit(x_train,y_train)


# 4. 예측 평가

score = model.score(x_test,y_test)
y_pred = model.predict(x_test)

print('model.socre :',score)
print('accuracy_score : ', accuracy_score(y_test,y_pred))
print('f1_score(macro) : ', f1_score(y_test,y_pred,average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_pred,average='micro'))
# print('f1_score(micro) : ', f1_score(y_test,y_pred))
# f1_score : 이진분류에서 사용 - 높으면 장땡 
# average='micro' 가중치 평균에 대한 계산방식으로  다중분류에서 사용할 수 있도록함 
# 다중분류에서 average 안쓰면 오류 난다. 
# fi_score는 통상 매크로를 많이 사용한다. 
# 재현율이 들어감 


'''
model.socre : 0.9487179487179487
acc :  0.9487179487179487
f1_score(macro) :  0.9439984430496765
f1_score(micro) :  0.9487179487179487
'''

print("======================SMOTE 적용 후 ==================================")
smote =SMOTE(
    random_state=3377, 
    k_neighbors= 5# 최근접 이웃 방식 | 디폴트 5
    # 엔개의 데이터 세트의 경우 엔의 제곱 샘플 쌍사이의 거리를 계산해야하며 이는 계산비용 이많이 든다. 
) 
# k 개 
# 데이터 중간즘에서 증폭으로 생성 
# 중복으로 증폭될 수 있다. 그것이 문제가 될 가능성이 잇다. 
# 거대한 단점 - 실무적인 단점 : 생성하는 시간이 엄청 걸린다. 
# 클래스 갯수의 제곱 
x_train, y_train= smote.fit_resample(x_train,y_train)
# 리 샘플링하겠다. 
print(x_train.shape,y_train.shape)
print(pd.Series(y_train).value_counts().sort_index())
# 클래스 중에서 값의 갯수가 적은 거 때문에 값이 쏠리는 현상을 줄이기 위해서 증폭하는 것이다.
# y_test는 검증이기 때문에 test는 증폭하지 않는다. 
# 변환된 데이터로 하는거라 별로 좋지는 않다. 의미도 없고 
#2-2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# 3-2. 훈련

model.fit(x_train,y_train)

# 4-2. 예측 평가

score = model.score(x_test,y_test)
y_pred = model.predict(x_test)

print('model.socre :',score)
print('accuracy_score : ', accuracy_score(y_test,y_pred))
print('f1_score(macro) : ', f1_score(y_test,y_pred,average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_pred,average='micro'))

