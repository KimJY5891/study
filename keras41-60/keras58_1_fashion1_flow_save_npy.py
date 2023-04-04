from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, accuracy_score
import time

start = time.time()
# 1. 데이터 
path = 'c:/study_data/_save/_npy/'
batch_size = 64
np.random.seed(333)
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
train_datagen=ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip = True,
    width_shift_range = 0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7
)
augment_size = 4000
np.random.seed(0)
randidx = np.random.randint(x_train.shape[0],size=augment_size)
print(randidx)
print(randidx.shape) # (100,)
print(np.min(randidx),np.max(randidx)) # 293 59352

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape,y_augmented.shape)
print(x_train.shape) 
x_train = x_train.reshape(-1, 28, 28, 1)
# (샘플 수, 28, 28, 1) -1은 샘플 수를 자동으로 계산하도록 지정하는데 
# 이는 원래 x_train의 샘플 수를 모르기 때문
print(x_train.shape) # (60000, 28, 28, 1)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_augmented = x_augmentsed.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)


print(x_train.shape,x_test.shape,x_augmented.shape) 
# (60000, 28, 28, 1) (10000, 28, 28, 1) (4000, 28, 28, 1)
print(np.min(x_train),np.max(x_train)) #0 255
print(np.min(x_test),np.max(x_test)) #0 255
print(np.min(x_augmented),np.max(x_augmented)) #0 255

# 변환
# x_augmented = train_datagen.flow(
#     x_augmented, y_augmented,
#     batch_size=batch_size,
#     shuffle=True
# ).next()[0] # 합치고

x_train = np.concatenate((x_train,x_augmented),axis=0)
y_train = np.concatenate((y_train,y_augmented),axis=0)
print(x_train.shape,x_augmented.shape)
np.save(path + 'keras58_1_fashion1_flow_save_x_train.npy',arr=x_train)
np.save(path + 'keras58_1_fashion1_flow_save_x_test.npy',arr=x_test)
np.save(path + 'keras58_1_fashion1_flow_save_y_train.npy',arr=y_train)
np.save(path + 'keras58_1_fashion1_flow_save_y_test.npy',arr=y_test)

end = time.time()
print('걸린 시간 : ', np.round(end-start,2))
