from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
# 1. 데이터 셋 
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data,columns=datasets.feature_names)
df['target'] = datasets.target
print(df)

# df.boxplot() # 이상치
# df.plot.box()
# 둘 다 같음 
# plt.show()

df.info()
print(df.describe())
# df['Population'].boxplot() # 컬럼 하나 뺀건 시리즈 라서 안됌 
# df['Population'].plot.box()
# 이건 됌 

df['Population'].hist(bins=50)
# 그래프가 한쪽으로 치우쳐있을 경우 
df['target'].hist(
    bins=50 # 범위
                  ) 
plt.show()
y= df['target']
x = df.drop(['target'],axis=1)
# =============================================================== xpopulation에 대한 로그변환 
x['Population'] = np.log1p(x['Population']) # 지수변환 exp1m(exp(지수변환) 1(일) m(마이너스) ) => np.exp1m 
# log1 = 0 
# log 0 => 연산 안됌 
# 로그 변환한다는건 
# 그래서 0문제 때문에 1을 더해준다. 
# print(x['Population'])
y = np.log1p(y)
# 분류형은 안된다. 
x_train , x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, train_size=0.8, random_state=337
)
# 이런식으로 왔다갔다하면 데이터 손상이 될 수도 있기 때문에

y_train_log = np.log1p(y_train)
y_test_log =  np.log1p(y_test)



# 2. 모델 
model = RandomForestRegressor(random_state=337)

# 3. 컴파일, 훈련 
model.fit(x_train,y_train_log)

# 4. 평가, 예측 
score = model.score(x_test,y_test)
print('로그 -> 지수r2 : ',r2_score(y_test,np.expm1(model.predict(x_test))))
# 변화전 : 
# score :  0.8021183994602941
# 변환후  
# x[pop]만 : score :  0.8022669492389668
# y만  :  0.8244322268075517
# 둘 다 : 
# 완전 치우쳐진 애들은 성능이 더 좋아진다. 
# 조심해야할 것 : 훈련을 트레인과 테스트로 만해서 
# 나중에 test데이터로 프레딕할때, 지수변환 해서 넣어줘야한다. 한다. 
x['Population'].hist()
plt.show  
