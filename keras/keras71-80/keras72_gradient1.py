import numpy as np
import matplotlib.pyplot as plt
f = lambda x : x**2 - 4*x +6

x = np.linspace(-1,6,100)
print(x,len(x))

y = f(x)
print(y)

plt.plot(x,y,'k-') 
plt.plot(2,2,'sk') 
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''
기울기가 0인 지점이 가장 낮은 지점 
기울기 0  = 작대기
'''
