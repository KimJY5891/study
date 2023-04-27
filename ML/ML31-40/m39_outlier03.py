import numpy as np
import pandas as pd
def outliers(data_out) :  # 이상치를 찾아주는 함수
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 :" , quartile_1)
    print("q2 :" , q2)             
    print("3사분위 :" , quartile_3)
    iqr = quartile_3 - quartile_1 
    print("iqr : ",iqr)
    lower_bound = quartile_1 - (iqr * 1.5) 
    upper_bound = quartile_3 + (iqr * 1.5) 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))


# 우리가 원하는 위치값이 나오게 수정해야한다. 
aaa = np.array([
    [-10,2,3,4,5,6,7,8,9,10,11,12,50],
    [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]
])
aaa = np.transpose(aaa)
aaa = pd.DataFrame(aaa,columns=["01","02"])

outliers_loc00 = outliers(aaa)
print(outliers_loc00)
# (array([6, 9], dtype=int64), array([1, 1], dtype=int64))

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

outliers_loc01 = outliers(aaa["01"])
outliers_loc02 = outliers(aaa["02"])
print(outliers_loc01)
print(outliers_loc02)

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.title('aaa["01"]')
plt.boxplot(aaa["01"])
plt.subplot(1,2,2)
plt.title('aaa["02"]')
plt.boxplot(aaa["02"])
plt.show()

