import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf 
tf.random.set_seed(337)
# 초기 가중치가 고정되지만, 연산하면서 고정된게 틀어진다. 
# 역전파하면서 값이 틀어진다. => 가중치 저장이 중요한 것 

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델
model = Sequential()
# [[0.94719875, 0.4854678 , 0.43610287]] 3노드의 임의 웨이트값 
model.add(Dense(3, input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

##########################################################
model.trainable = True # 디폴트
##########################################################
model.summary()

model.compile(loss='mse',optimizer = 'adam')
model.fit(x,y,batch_size=1, epochs = 50 )
y_pred = model.predict(x)
print(y_pred)
# model.trainable = False로 해놓으면 로스가 갱신이 안된다.  


