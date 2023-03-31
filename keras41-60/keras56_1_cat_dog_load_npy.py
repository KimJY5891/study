import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
start= time.time()

path = 'c:/study_data/_data/cat_dog/PetImages/'
save_path = 'c:/study_data/_save/cat_dog/'

x_train = np.load(path+'keras56_1_x_train.npy')
y_train = np.load(path+'keras56_1_y_train.npy')
x_test = np.load(path+'keras56_1_x_test.npy')
y_test = np.load(path+'keras56_1_y_test.npy')

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(200,200,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
hist = model.fit(
    x_train,y_train,
    epochs=10, batch_size=16,
    validation_data=[x_test,y_test],
    validation_steps=24,
)# (160,100,100,1)
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
print('loss : ',loss[-1]) 
print('val_loss : ',val_loss[-1]) 
print('acc : ',acc[-1]) 
print('val_acc : ',val_acc[-1])



# 4. 평가 예측

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.plot(loss)
plt.subplot(2, 2, 2)
plt.plot(val_loss)
plt.subplot(2, 2, 3)
plt.plot(acc)
plt.subplot(2, 2, 4)
plt.plot(val_acc)
plt.suptitle('Training Result') 
plt.show()
end= time.time()
print('걸리는 시간 : ',np.round(end-start,2))
