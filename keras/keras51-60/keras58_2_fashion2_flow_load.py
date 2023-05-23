import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score


# 1. 데이터

path = 'c:/study_data/_save/_npy/'
x_train = np.load(path + 'keras58_1_fashion1_flow_save_x_train.npy')
x_test = np.load(path + 'keras58_1_fashion1_flow_save_x_test.npy')
y_train = np.load(path + 'keras58_1_fashion1_flow_save_y_train.npy')
y_test = np.load(path + 'keras58_1_fashion1_flow_save_y_test.npy')
print(x_train.shape, x_test.shape) # (64000, 28, 28, 1) (10000, 28, 28, 1)
print(y_train.shape, y_test.shape) # (61000,) (10000,)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(32,(2,2),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='softmax'))


# 3. 컴파일, 훈련 

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
hist = model.fit(
    x_train,y_train,
    epochs=5, batch_size=16,
    validation_data=[x_train,y_train],
    validation_steps=24
)
print(hist)

# 4. 평가 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
y_predict=np.argmax(model.predict(x_test),axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

import matplotlib.pyplot as plt
plt.figure(figsize=(2,10))
for i in range(9) : 
    plt.subplot(2,1,i+1)
    plt.axis('off')
    plt.imshow(x_train,cmap='gray')
    plt.subplot(2,10,i+11)
    plt.imshow(x_train,cmap='gray')
plt.show()
