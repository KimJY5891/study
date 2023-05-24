import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
#     return np.maximum(0,x)
# 엑스와 영 중에서 엑스가 마이너스면 영으로 나오고 엑스가 100이고 0이 0 이면 100이 나옴 

relu = lambda x : np.maximum(0,x)
x = np.arange(-5,5,0.1)
y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()

