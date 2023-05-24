import numpy as np
import matplotlib.pyplot as plt
# def f(x) :
#     if x > 0:
#         return scale * x
#     else:
#         return scale * alpha * (exp(x) - 1)
# 일반적으로 alpha 값은 1.6733
# scale 값은 일반적으로 1.0507로 설정됩니다. 
scale = 1.0507
alpha = 1.6733
# def selu(x,scale,alpha) :
#     return np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))

selu = lambda x, scale, alpha : np.where(x>0,scale*x, scale * alpha * (np.exp(x) - 1))
x = np.arange(-50000,50000,15)
y = selu(x,scale,alpha)

plt.plot(x,y)
plt.title('selu')
plt.grid()
plt.show()
