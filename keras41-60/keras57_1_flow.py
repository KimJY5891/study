# 이제까지는 이미지를 포러데엇 가져다가 수치화 시켰다. 
# 하지만 수치로 제공되는 이미지 데이터도 잇다. 
# 그것을 우리는 증폭을 할 것이다. 
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()
train_datagen = ImageDataGenerator(
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
augment_size = 100, # 
print(x_train.shape) #(60000, 28, 28)
print(x_train[0].shape) # (28, 28) 
print(x_train[1].shape) # (28, 28)
print(x_train[0][0].shape) # (28,)

print(np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1)
# print(np.tile(x_train[0].reshape(28*28) - > 100r개로 늘림 
#              ,augment_size).reshape(-1,28,28,1) -> 다시 원래 사이즈로 바꿔줌 
#       .shape)
# tile - 동일하게 생긴 애들을 게속 복붙 시키는 것이다. 
# np.tile(데이터, 증폭시킬 갯수)
# reshape(-1,28,28,1)에 -1이란 전체 행을 말한다. 
print(np.zeros(augment_size)) 
print(len(np.zeros(augment_size))) # 100
print(np.zeros(augment_size).shape) # (100,)
x_data= train_datagen.flow(
    # 처음에 경로가 아닌 데이터를 받는다. 
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),# x 데이터 
    np.zeros(augment_size), # y데이터 : 그림만 그릴거라 필요없어서 걍 0 넣어줘서 
    batch_size=augment_size,
    shuffle=True
)

# print(x_data) # 이터레이터 : <keras.preprocessing.image.NumpyArrayIterator object at 0x0000025298CC5DF0>
# print(x_data[0]) # x와 y가 모두 포함 
# print(x_data[0][0].shape) #x데이터 (100, 28,28,1)
# print(x_data[0][1].shape) #y데이터 (100,)
# #flow_from_directory - 디렉토리에 있는 파일로  xy만듦
# # flow - 원래있는 데이터로 증폭하는 것 

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49) : 
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i],cmap='gray')
plt.show()


# 각종 키워드 이해하기 

# 월요일에 수정 더 함 

