  import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
# np.save(path + '파일명',arr=???)
path = 'c:/study_data/_data/cat_dog/PetImages/'
#path = 'd:/study_data/_data/cat_dog/PetImages/'

#1. 데이터 
start= time.time()
datagen=ImageDataGenerator(
    rescale=1./255,
#     horizontal_flip=True, 
#     vertical_flip=True ,
#     width_shift_range=0.1,
#     height_shift_range=0.1, 
#     rotation_range=5, 
#     zoom_range=1.2, 
#     shear_range= 0.7,
#     fill_mode='nearest'
)
datasets = datagen.flow_from_directory(
    path,
    target_size= (200,200), 
    batch_size=5000, 
    class_mode='binary', 
    color_mode='rgb', 
    shuffle=False 
)   # Found 25000 images belonging to 2 classes.
    # <keras.preprocessing.image.DirectoryIterator object at 0x0000018D43430B20>

# print(len(xy))    # 
# print(len(xy[0])) # 
# print(xy[0][0]) # x 전체
# print(xy[0][1]) # y 전체
x = datasets[0][0]
y = datasets[0][1]

print('datasets : ',np.array(datasets))
'''

print('y : ',y)
print('len(x) : ',len(datasets[0][1]))
print('len(x) : ',len(datasets[0][0]))
print('len(x) : ',len(datasets[1][0]))
print('len(x) : ',len(datasets[1][1]))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.1,random_state=333, shuffle=False
)
print("x : ",x.shape)
print("y : ",y.shape)
print("datasets : ", datasets.shape)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


np.save(path+'keras56_1_x_train.npy',arr=x_train)
np.save(path+'keras56_1_y_train.npy',arr=x_test)
np.save(path+'keras56_1_x_test.npy',arr=y_train)
np.save(path+'keras56_1_y_test.npy',arr=y_test)
end= time.time()
print('걸리는 시간 : ',np.round(end-start,2))


'''
