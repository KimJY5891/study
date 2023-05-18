from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time 

np.random.seed(333)

# 1. 데이터 
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

randidx = np.random.randint(x_train.shape[0],size=augment_size)
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
start= time.time()

# 증폭
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, 
    batch_size=augment_size,
    shuffle=True,
    save_to_dir='c:/temp/'
    #save_to_dir='d:/'
).next()[0]
# 증폭하는 부분에서 세이브 
end= time.time()
print(augment_size,'개 걸린시간 : ',round(end-start,2),'초')
# temp : 임시적인 -> 임시파일

