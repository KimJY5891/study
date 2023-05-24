# vgg가 전이학습 기초
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16,VGG19,DenseNet121,ResNet101
# 텐서플로우에서 제공하는건 한정적임
# vgg16 학습용이지만 실무에서도 사용 가능하다.\

vgg16  = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3)
               )
vgg16.trainable = False # r가중치 동결 
print(len(vgg16.weights)) # 26 -> 26
print(len(vgg16.trainable_weights)) # 26 -> 0

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()
# model.trainable = True
# false여도 
model.summary
print(len(model.weights)) # 30 -> 30 -> 가중치 동결 후 30
print(len(model.trainable_weights)) # 30 -> 0 -> 가중치 동결 후 4
'''
Total params: 14,766,998
Trainable params: 52,310
Non-trainable params: 14,714,688

전이학습이 동결시키는게 좋을 수도 있고 학습시키는게 좋은 걸 수도있다. 
'''
