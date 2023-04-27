import numpy as np
aaa = np.array([-10,2,3,4,5,6,700,8,9,10,11,12,50]) # 13개 , 중위값 7
# aaa = aaa.reshape(-1,1)
# 리쉐이프 한 이유 : 벡터 형태로 안받기 때문에
from sklearn.covariance import EllipticEnvelope
# covariance(공(공공의)분산)

outliers = EllipticEnvelope(contamination=.3)
# contamination : 전체데이터 중에 몇프로를 이상치로 할 것인가 ? 
# .1 = 10 %, .2 = 20%
outliers.fit(aaa)
result = outliers.predict(aaa)
print(result)
# 위치를 이용해서 이상치를 받아서 활용할 수 있다. 
# -1 : 이상치 
