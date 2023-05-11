import numpy as np
import matplotlib.pyplot as plt

data= np.random.exponential(scale=2.0,size = 1000)
# exponential : 기하급수적

# 로그 변환 
log_data = np.log(data)
# 지수 2 -> 로그 100
# 원본 데이터 히스토그램 그리자 
plt.subplot(1,2,1)
plt.hist(data, bins=50, color = 'blue',alpha = 0.5)
plt.title('original')
# 
plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color ='red',alpha = 0.5)
plt.title('Log Transformed Data')

plt.show()

# 어떻게 가운데로 모으는 것이 우리의 실력이 된다. 
