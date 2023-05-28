import numpy as np
from tensorflow.keras.datasets import cifar10


# 1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print("train : ",x_train.shape,y_train.shape) #train :  (50000, 32, 32, 3) (50000, 1)
print("test : ",x_test.shape,y_test.shape) #test :  (10000, 32, 32, 3) (10000, 1)
print("x_train :",x_train[0])
print("y_train :" ,y_train[0])


import matplotlib.pyplot as plt
plt.imshow(x_train[0],'rgb')
plt.show()
