# 난 정말 시그모이드 구현~ 
import numpy as np
import matplotlib.pyplot as plt

sigmoid = lambda x : 1/(1+np.exp(-x))
x = np.arange(-5,5,0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.grid()
plt.show()

# 같은 비율로 조정하는게 스케일링 개념이고 
# 쓸모없는 비율은 컷해버리는게 활성화함수 
# def sigmoid(x) : 
#     return 1/(1+np.exp(-x))

