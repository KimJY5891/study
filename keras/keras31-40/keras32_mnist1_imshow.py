import numpy as np
from tensorflow.keras.datasets import mnist 
(x_train,y_train),(x_test,y_test) = mnist.load_data() # 데이터를 넣어줌 
#print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape)) # 흑백은 1이라 굳이 나오지 않았음
#print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape)) # cnn 모델 사용할 대는 reshape 이용해서 (60000,28,28,1)로 만들어줘야함

print(x_train[0])
print(y_train[3333])

import matplotlib.pyplot as plt
plt.imshow(x_train[3333],'gray') # 그림 보여줌
plt.show()
