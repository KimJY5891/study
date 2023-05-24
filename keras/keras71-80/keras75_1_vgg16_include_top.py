# vgg 가 전이학습 기초
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16,VGG19,DenseNet121,ResNet101
# 텐서플로우에서 제공하는건 한정적임
# vgg16 학습용이지만 실무에서도 사용 가능하다.

# model  = VGG16() # include_top=True, input_shape=(224,224,3)
model  = VGG16(
    weights='imagenet',
    # include_top=True,
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(32, 32, 3)
    include_top=False,
    input_shape=(32,32,3)
               )# 이미지넷에서 사용했던 가중치를 사용해서 쓸거야 
# 플리커넥티드 레이어 : 밀집모자 구조 찾아보기 
# 인풋과 아웃풋빼고 나머지 부분의 가중치만 사용하는 것이다. 
# 
model.summary()
print(len(model.weights)) # 32 -> 26
print(len(model.trainable_weights)) # 32 -> 훈련 가능하다는 이야기
# 32 -> 26
#################### include_top = True #####################\
# 1. FC layer 원래 것 사용
# 2. input_shape=(224,224,3) 고정값 - 바꿀 수 없다. 
# 3. Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792

#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928

#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856

#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584

#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168

#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080

#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160

#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808

#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808

#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000   

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________
# 32
# 32
#################### include_top = false #####################\
# 1. FC layer 원래 것 삭제 -> 나중에 커스터 마이징
# 2. input_shape(224,224,3) 
# Model: "vgg16"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0

#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792

#  block1_conv2 (Conv2D)       (None, 32, 32, 64)        36928

#  block1_pool (MaxPooling2D)  (None, 16, 16, 64)        0

#  block2_conv1 (Conv2D)       (None, 16, 16, 128)       73856

#  block2_conv2 (Conv2D)       (None, 16, 16, 128)       147584

#  block2_pool (MaxPooling2D)  (None, 8, 8, 128)         0

#  block3_conv1 (Conv2D)       (None, 8, 8, 256)         295168

#  block3_conv2 (Conv2D)       (None, 8, 8, 256)         590080

#  block3_conv3 (Conv2D)       (None, 8, 8, 256)         590080

#  block3_pool (MaxPooling2D)  (None, 4, 4, 256)         0

#  block4_conv1 (Conv2D)       (None, 4, 4, 512)         1180160

#  block4_conv2 (Conv2D)       (None, 4, 4, 512)         2359808

#  block4_conv3 (Conv2D)       (None, 4, 4, 512)         2359808

#  block4_pool (MaxPooling2D)  (None, 2, 2, 512)         0

#  block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808

#  block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0
# 풀래튼 하단 부분(플리커넥티드 레이어 부분)
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# 26
# 26
