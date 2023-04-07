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

randidx = np.random.randint(x_train.shape[0],size=augment_size)
# 랜덤하게 육만개에서 4만개 뽑겟다.
print(randidx) #[43351 14109 56941 ... 12193 12719 15038]
print(randidx.shape) # (40000,)
print(np.min(randidx),np.max(randidx)) # min : 0 59998
 
x_augmented = x_train[randidx].copy() #(40000, 28, 28)
y_augmented = y_train[randidx].copy() # 넘파이

print(x_augmented) 
print(x_augmented.shape,y_augmented.shape) #(40000,28,28) (40000,)
#4만개의 중복을 방지하기 위해서 .copy()해야한다.
#x_augmented 아마도 나중에 많이 바꿀지도??

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
'''
# 1. 변환

x_augmented = train_datagen.flow(
    x_augmented, y_augmented, 
    batch_size=augment_size,
    # 통사이즈
    shuffle=True
)
#x_augmented의 형태는 이터레이터 형태 
print(x_augmented)
#<keras.preprocessing.image.NumpyArrayIterator object at 0x000002921DFDDDF0>
# 넥스트를 사용하지 않는다면 통배치니까 
print(x_augmented[0][0].shape)#(40000, 28, 28, 1) 
print(x_augmented[0][1].shape)#(40000,)
'''
# 2. 넥스트로사용 방법 

x_augmented = train_datagen.flow(
    x_augmented, y_augmented, 
    batch_size=augment_size,
    # 통사이즈
    shuffle=True
).next()[0]
# next()[0]을 해야 x_augmented[0][0]이 나온다.
print(x_augmented)
print(x_augmented.shape) #(40000, 28, 28)

# x_augmented과 x_train 합체
# 에러로 찾는 방법 
# 넘파이 행렬로 찾는 방법

print(np.min(x_train),np.max(x_train))#255. 0. 
print(np.min(x_augmented),np.max(x_augmented))# 1. 0.
print(x_train.shape) #(60000, 28, 28, 1)
print(x_augmented.shape) #(40000, 28, 28, 1)
# 아규먼티드는 스케일링이 되어있고 x트레인과 엑스 테스트은 스케일링이 안되어잇다. 
# 와이는 스케일링하면 안되어있다.


x_train = np.concatenate((x_train, x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)
#사슬 처럼 엮다.

print(x_train) 
print(x_train.shape)
'''

'''
