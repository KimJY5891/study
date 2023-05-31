import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler #전처리
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import r2_score, accuracy_score


# 1. 데이터 
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape)) #((60000, 28, 28), (60000,)) ((10000, 28, 28), (10000,))


import matplotlib.pyplot as plt
plt.imshow(x_train[0],'rgb') # 그림 보여줌
plt.show()
