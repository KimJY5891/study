# 모델의 성능을 보고 좋은지 아닌지 판단하기 
# 디폴트 값이 엉망이라면 피처인포턴스를 신뢰할 수 없다. 
# 모델이 성능이 좋았을 경우, 나온거에 대한 피처 인포턴스가 정확하다.
# 4개의 모델 중에서 최상의 값에서 피처인포턴스를 돌려야한다. 
# 그러고 나서 컬럼을 삭제하고 다시 돌려야한다. 
# 하위 20~25% 컬럼만 삭제하는게 아니라 성능보고 삭제하는게 가장 좋다.

# 두 개의 컬럼이 같은 가중치 값이 같다면 같은 선을 그리고 있다. 
# 풍속 풍향이 가중치가 비슷한 컬럼이 좀 더 종속적으로 변할 수도 있다. 
# 비슷한 놈은 하나를 제거함
# 그래서 5:5 엔빵이 된다.
# 성능이 좋을 지는 해봐야 안다. 

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. 데이터 

datasets = load_iris()
print(datasets.feature_names) # 사이킷런에서 제공하는 데이터만 가능
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# 판다스에서 columns

x = datasets['data']
y = datasets['target']
# 넘파이로 되어 있는 데이터 판다스로 넣기 
df = pd.DataFrame(x,columns=datasets.feature_names)
# print(df)

df['target(Y)'] = y
# 새로만들 컬럼이름 = 값을 넣어주면 된다. 
print(df) # [150 rows x 5 columns]

print("================================상관계수 히트 맵 짜잔 ===================================")
print(df.corr())
#  correlation - 상관
'''
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target(Y)
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

단순회귀라서 100프로 믿으면 안된다. 
단순 리니어 
상관관계 선이라고 생각하면 됌 
y와의 상관관계가 높은 애를 봐야한다.
그렇다고 단순리니어라서 다 맞다고 보기는 어렵다. 
target(Y) - 높은 것이 두개가 있는데,  두개가 있다면 하나는 쳐내야 할지도 ~ ?
petal length (cm) -   petal width (cm) 둘이 0.96 이라면 둘 중 하나를 쳐내야 할지도~ ?
'''

import matplotlib.pyplot as plt
import seaborn as sns # 왜 sns일까 관계성이 없이 그냥 sns으로 사람들이 많이 사용해서 이렇게 굳어진게 된 것 
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(),square=True, annot=True, cbar=True)
plt.show()
# 피쳐인포턴스 와 corr 중에서 하나 선택해서 보고 쳐내보기 
