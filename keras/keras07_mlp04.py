import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([range(10),range(21,31),range(201,211)])
print(x)
#[  0   1   2   3   4   5   6   7   8   9]
#[ 21  22  23  24  25  26  27  28  29  30]
#[201 202 203 204 205 206 207 208 209 210]
print(x.shape) #(3,10)
x=x.T
print(x.shape)#(10,3)
y=np.array([[1,2,3,4,5,6,7,8,9,10]]) #(1,10)
y=y.T
print(y)
# 2. 모델 구성
# 3. 컴파일, 훈련
# 4. 평가, 예측
