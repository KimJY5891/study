from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
augment_size = 100
print(x_train.shape) 
print(x_train[0].shape) 
print(x_train[1].shape) 
print(x_train[0][0].shape) 

print(np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1)

print(np.zeros(augment_size)) 
print(len(np.zeros(augment_size))) 
print(np.zeros(augment_size).shape) 
x_data= train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),# x 데이터 
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=True
).next()
#########################. next()미사용 #########################
print('x_data : ', x_data) # x와 y가 합쳐진 데이터 출력
# 첫 번쩨인 x_data[0]의 값이 나옴 
print(type(x_data)) # <class 'tuple>
print(x_data[0]) # x데이터
print(x_data[1]) # x데이터
print(x_data[0].shape,x_data[1].shape) 
print(type(x_data[0]))
# 튜플 (x,y) - 이터레이터
# 둘 다 넘파이 
#########################. next()미사용 ######1###################
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49) : 
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i],cmap='gray') #.next() 미사용
    #plt.imshow(x_data[0][0][i],cmap='gray') #.next() 미사용 (x_data[0][0][i] -> 배치가 있어서 이런 식으로 작성했다.  [배치 ][?][ ?]
    #  x_data[0][i] - 통배치는 앞에 하나만 줄어들면 된다.  
plt.show()


