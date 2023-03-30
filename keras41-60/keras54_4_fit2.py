import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
# fit으로 사용할거면 flow from 디렉토리로 던질 필요가 없다. 
#
#1. 데이터 

traindatagen=ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    vertical_flip=True ,
    width_shift_range=0.1,
    height_shift_range=0.1, 
    rotation_range=5, 
    zoom_range=1.2, 
    shear_range= 0.7,
    fill_mode='nearest' 
)
testdatagen=ImageDataGenerator(
    rescale=1./255,
)
xy_train = traindatagen.flow_from_directory(
    'c:/study_data/_data/brain/train/',
    #'d:/study_data/_data/brain/train/',
    target_size= (100,100), 
    batch_size=5, 
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
) 
xy_test = testdatagen.flow_from_directory(
    'c:/study_data/_data/brain/test/',
    #'d:/study_data/_data/brain/test/',
    target_size= (100,100), 
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
) 

 
# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2),padding='same',input_shape=(100,100,1),activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
# hist = model.fit(
#     # model.fit_generator으로 통배치로 바꿔주면된다. 
#     xy_train[0][0],xy_train[0][1],
#     epochs=10, batch_size=16,
#     validation_data=(xy_test[0][0],xy_test[0][1])
# )# (160,100,100,1)
# hist = model.fit_generator(
#     xy_train,epochs=30,
#     steps_per_epoch=32, 
#     validation_data=xy_test,
#     validation_steps=24, )
hist = model.fit(
    # model.fit_generator으로 통배치로 바꿔주면된다. 
    xy_train,
    epochs=10, batch_size=16,
    validation_data=xy_test,
    validation_steps=24,
)# (160,100,100,1)
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
print('loss : ',loss[-1]) 
print('val_loss : ',val_loss[-1]) #
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
