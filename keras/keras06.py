import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x1=np.array([
    [1,2],
    [3,4]
]) #(2,2)
x2=np.array([
    [[1,2,3]]
])#(1,1,3)
x3=np.array([
    [[1,2,3],[4,5,6]]
])#(1,2,3)
x4=np.array([
    [1],
    [2],
    [3]
])(3,1)
x5=np.array([
    [[1]],
    [[2]],
    [[3]]
])(1,3,1)
x6=np.array([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])#(2,2,2)
x7=np.array([
    [[1,2]],
    [[3,4]],
    [[5,6]],
    [[7,8]]
])#(4,1,2)
print(x1.shape)#(2,2)
print(x2.shape)#(1,1,3)
print(x3.shape)#(1,2,3)
print(x4.shape)#(3,1)
print(x5.shape)#(3,1,1)
print(x6.shape)#(2,2,2)
print(x7.shape)#(4,1,2)
