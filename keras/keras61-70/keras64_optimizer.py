# 아이리스 생략 
import numpy as np

# 1. 데이터 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection  import train_test_split

datasets =fetch_california_housing()
x= datasets.data
y= datasets.target
x_train,x_test, y_train,y_test  = train_test_split(
    x,y, train_size= 0.8, stratify=y, random_state=337
)

# 2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Dense

model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

# 3. 컴파일 훈련 
# model.compile(loss='mse ',optimizer='adam') # 러닝레이트 디폴트 값 찾기 
from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate=learning_rate) # Adam optimizer의 learning_rate 디폴트 값 =  0.001 
model.compile(loss='mse ',optimizer=optimizer) 
'''
2번째 방법
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # 러닝레이트를 감축시켜라 
es = EarlyStopping(monitor = 'val_loss',patience = 20, mode = 'min', verbose = 1)
rlr = ReduceLROnPlateau(monitor='val_loss', patience= 10,# 10까지 개선이 없으면 러닝레이트를 줄여버리겠다.
                        mode= 'auto',verbose=1, factor=0.5 # 반띵 해주기
                       )# EarlyStopping이랑 ReduceLR이 같이 사용할 때 수가 같으면 ReduceLR이 안 먹힌다. 먹히기 전에 EarlyStopping으로 끝나기 때문이다.  
# 덩어리가 더 큰 놈들이 잘 먹힌다.
model.fit(x_train, y_train, epochs=10, batch_size = 32,verbose = 1, validation_split=0.2, callbacks=[es,rlr])
'''
# 4. 평가, 예측

result = model.evaluate(x_test,y_test)  
y_pred = model.predict(x_test)
print('acc : ',result[1])


