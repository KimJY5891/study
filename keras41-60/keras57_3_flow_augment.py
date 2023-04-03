from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
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
# print(x_train.shape) #(60000, 28, 28)
# print(x_train[0].shape) # (28, 28) 
# print(x_train[1].shape) # (28, 28)
# print(x_train[0][0].shape) # (28,)
#randidx = np.random.randint(60000, size = 40000)
randidx = np.random.randint(x_train.shape[0],size=augment_size)
# 랜덤하게 육만개에서 4만개 뽑겟다.
print(randidx) #[43351 14109 56941 ... 12193 12719 15038]
print(randidx.shape) # (40000,)
print(np.min(randidx),np.max(randidx)) # min : 0 59998
# 육만개 중에서 이걸 뽑아서 증폭시키겠다. 
# 시드가 바뀌면 성능이 바귄다. 
# 넘파이 시드 고정을 찾아야한다. 
x_augmented = x_train[randidx] #(60000, 28, 28)
# 4만개 들어감 
y_augmented = y_train[randidx] # 
print(x_augmented) 
print(x_augmented.shape,y_augmented.shape) #(40000,28,28) (40000,)
