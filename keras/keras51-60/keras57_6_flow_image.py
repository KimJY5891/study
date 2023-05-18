
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터 

np.random.seed(333)
(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True, 
    #vertical_flip=True ,
    width_shift_range=0.1,
    height_shift_range=0.1, 
    rotation_range=5, 
    zoom_range=1.2, 
    shear_range= 0.7,
    fill_mode='nearest' 
) 
augment_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augment_size)
# 랜덤하게 육만개에서 4만개 뽑겟다.
print(randidx) #[43351 14109 56941 ... 12193 12719 15038]
print(randidx.shape) # (40000,)
print(np.min(randidx),np.max(randidx)) # min : 0 59998
x_augmented = x_train[randidx].copy() #(40000, 28, 28)
y_augmented = y_train[randidx].copy() #
print(x_augmented) 
print(x_augmented.shape,y_augmented.shape) #(40000, 28, 28) (40000,)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
print(x_train.shape,x_test.shape,x_augmented.shape)
#(60000, 28, 28, 1) (10000, 28, 28, 1) (40000, 28, 28, 1)

# 변환 
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, 
    batch_size=augment_size,
    shuffle=True
).next()[0]
print(np.min(x_train),np.max(x_train)) #0 255
print(np.min(x_augmented),np.max(x_augmented)) #0 255
print(np.min(x_test),np.max(x_test)) #0 255
print(x_train.shape,x_augmented.shape) #(60000, 28, 28, 1) (40000, 28, 28, 1)
 
x_train=np.concatenate((x_train,x_augmented),axis=0)
y_train=np.concatenate((y_train,y_augmented),axis=0)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)  #[0. 0. 0. ... 0. 0. 1.]
print(y_train.shape) #(100000, 10)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(32,(2,2),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

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

result=model.evaluate(x_test,y_test)
print('result : ',result)
y_pred = np.argmax(model.predict(x_test),axis=1)
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test,y_pred)
print('acc : ',acc)

import matplotlib.pyplot as plt
plt.figure(figsize = (7, 7))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    #plt.subplot(nrows, ncols, index)
    plt.axis('off')
    plt.imshow(x_train[i + 60000], cmap = 'gray')
    # cmap = colormap
    plt.subplot(2, 10, i + 11)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap = 'gray')
plt.show()
