import numpy as np
import matplotlib.pyplot as plt

# def elu(x,alp)   : 
#     return (x>0)*x + (x<=0)*(alp*(np.exp(x)-1))

elu = lambda x, alp : (x>0)*x + (x<=0)*(alp*(np.exp(x)-1))
alp = 0.5
x = np.arange(-5,5,0.1)
y = elu(x,alp)

plt.plot(x,y)
plt.grid()
plt.show()
