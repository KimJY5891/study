# 수정요망
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 다중공선
# 컬럼 마다 선을 그어서 상태를 판단하겠다는 것 
# 궁극적인 목표는 y값을 찾아내는 것 
# 상관관계가 높은 몇 개의 컬럼 때문에 과적합이 걸릴 수도 있다. 
# 상관관계가 너무 높을 경우 하나를 삭제하자 
# 코릴레이션과 비슷하다. 
# 두 개의 컬럼이 상관도가 너무 높다. -> 제거 혹은 차원 축소 
# 무조건 제거가 좋은 것이 아니다. 살려보고 싶을 경우 차원축소도 가능하다. 
# 코릴레이션과 같이 사용하면 된다. 
data={'size':[30,35,40,45,50,45],
      'rooms' : [2,2,3,3,4,3],
      'window' : [2,2,3,3,4,3],
      'year':[2010,2015,2010,2015,2010,2014],
      'price' : [1.5,1.8,2.0,2.2,2.5,2.3],
}

df = pd.DataFrame(data)
print(df)

x = df[['size','window','rooms','year']]
y = df['price']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled)

vif = pd.DataFrame()
vif['variables'] = x.columns
vif['vif'] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
print(vif)
'''
  variables         vif
0      size  378.444444
1    window         inf
2     rooms         inf 두개의컬럼이 완전히 똑같다는 것 
3      year   53.333333
'''
# for의 값이 aaa에 들어간다. 
# for i in range(x_scaled.shape[1]) : # 컬럼의 갯수
# 19이하일 대 다중 공성성이 높지 않다고 판단한다. 
# 다 10초과로 높을 때 하나씩 줄여서 판단하기

print("================== rooms 제거 전 =====================")
lr = LinearRegression()
lr.fit(x_scaled,y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y,y_pred)
print('r2:',r2) # r2: 0.9938931297709924

print("================== rooms 제거 =====================")
x_scaled = df[['size','window','year']]
vif2 = pd.DataFrame()
vif2['variables'] = x_scaled.columns
vif2['vif'] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]

lr.fit(x_scaled,y)
print(vif2)
lr.fit(x_scaled,y)
y_pred = lr.predict(x_scaled)
r2 = r2_score(y,y_pred)
print('r2:',r2)
'''
  variables         vif
0      size  295.182375
1    window  139.509263
2      year   56.881874
r2: 0.9938931297709941
'''
# ValueError: Length of values (2) does not match length of index (3) 컬럼이 두 개라서 다중공선성 계산이 안된다. 
print("================== 사이즈 제거 =====================")

print("================== year 제거 =====================")


