import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid,tanh,swish

# def custom_swish(x):
#     return x*(1/(1+np.exp(-x)))
custom_swish = lambda x: x*(1/1+np.exp(-x))
x=np.arange(-5,5,0.1)

plt.subplot(1,2,1)
plt.plot(x,swish(x),label='keras swish')
plt.title('keras swish')
plt.subplot(1,2,2)
plt.plot(x,custom_swish(x),label='custom swish')
plt.title('custom swish')
plt.legend()
plt.show()
