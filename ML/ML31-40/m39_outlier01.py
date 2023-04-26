import numpy as np
import pandas as pd 
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50]) # 13개 , 중위값 7 
# 섞어도 위치는 찾아준다. 
# 아래는 그 위치에 있는 값으로 찾아준다. 
def outliers(data_out) :  # 이상치를 찾아주는 함수
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 :" , quartile_1) # 4. # 14개 의 4분의 1이니가 3.5라서 4번째 내놓음
    print("q2 :" , q2)              # 7.
    print("3사분위 :" , quartile_3) # 10.
    iqr = quartile_3 - quartile_1 # 6
    print("iqr : ",iqr)
    lower_bound = quartile_1 - (iqr * 1.5) #  4 - 9 = -5
    upper_bound = quartile_3 + (iqr * 1.5) # 10 + 9 = 19 # 1.5가 가장 좋다고 학자들이 말함 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
    # 19보다 크거나, -5보다 작은 넘
    # 위치 값을 반환
    # np.where = if 문 비슷
outliers_loc = outliers(aaa)
print('이상치의 위치 : ',outliers_loc)
print(np.percentile(aaa,[25])) # 4. 
print(np.percentile(aaa,[50])) # 7. 
print(np.percentile(aaa,[75])) # 10. # 데이터의 75%로 지점은 정상적이지 않을까 싶어서 지정
# 정상범위가 중간값에서 1.5배 사이에 정상값이 있지 않을까???

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()
# 박스 안에 있는거만 이상치가 아니라고 생각 
# 가운데 줄이 중위값 
# 위에가 3사분위
# 밑에가 1사분위

aaa = np.array([-10,2,3,4,5,6,7,8,1000,9,10,11,12,50]) # 14개 , 중위값 7.5
#  4.25가 1사분위 
# 선형 계산을 해서 계산을 한다. 
aaa = np.array([-10,2,3,4,5,6,7,8,1000,9,10,11,12,50,333]) # 15개 , 중위값 8
# 4.5 1사분위

