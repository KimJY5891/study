import numpy as np
import matplotlib.pyplot as plt
# def leaky_relu(x) : 
#     max(0.01*x,x)
leaky_relu = lambda x : max(0.01*x,x)

x = np.arange(-128,48,5)
y = leaky_relu(x)

plt.plot(x,y)
plt.title('leaky relu')
plt.grid()
plt.show()
