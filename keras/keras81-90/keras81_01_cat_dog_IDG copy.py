# 최종지표인 acc 도출
# 기존 것과 전이학습의 성능비교 
# 무조건 전이 학습이 이겨야한다. 
# 본인 사진 넣어서 개인지 고양인지 구별

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
from tensorflow.keras.utils import to_categorical
import time
from pprint import pprint
#1. 데이터 

save_path = 'd:/study_data/_save/_npy/cat_dog/'
path = 'd:/study_data/_data/cat_dog/PetImages/'
pred_path = 'd:/study_data/_data/cat_dog/pred'

stt = time.time()

datagen  =ImageDataGenerator(rescale=1./255)
x_pred_datagen = ImageDataGenerator(rescale=1./255)

xy = datagen.flow_from_directory(
    path,
    target_size= (200,200), 
    batch_size=5,
    class_mode='binary',
    color_mode='rgb', 
    shuffle=True
)

x_pred = x_pred_datagen.flow_from_directory(
    pred_path,
    target_size= (200,200), 
    batch_size=1000,
    color_mode='rgb', 
    shuffle=True
)
pprint('x_pred : ', x_pred )

# 
print('xy[0]:',xy[0])
# xy[0]은 배치 크기만큼의 데이터 세트를 나타내며,
# xy[0][0]은 해당 배치의 입력 데이터입니다. 즉 수치화된 이미지
print('xy[1]:',xy[1])


x = xy[0][0]
y = xy[0][1]

print(y,y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=640, test_size=0.1)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

np.save(save_path + 'x_train.npy', arr=x_train)
np.save(save_path + 'x_test.npy', arr=x_test)
np.save(save_path + 'y_train.npy', arr=y_train)  
np.save(save_path + 'y_test.npy', arr=y_test) 
np.save(save_path + 'x_pred.npy', arr=x_pred) 

ett1 = time.time()
print('넘파이 변경 시간 :', np.round(ett1-stt, 2)) 
 
