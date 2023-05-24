import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input
from tensorflow.keras.applications import VGG16,VGG19,DenseNet121,ResNet101,InceptionV3
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape  = (32,32,3))
print(base_model.output)

x = base_model.output # 최종모델의 아웃풋을 보여줌
# KerasTensor(type_spec=TensorSpec(
# shape=(None, None, None, 512),# 전해주기 직전의 쉐이프
# dtype=tf.float32, name=None), name='block5_pool/MaxPool:0', description="created by layer 'block5_pool'")
x = GlobalAveragePooling2D()(x)
output1 = Dense(10,activation='softmax')(x)
model = Model(inputs = base_model.input, outputs= output1)

'''
result :  0.7713070511817932
313/313 [==============================] - 2s 6ms/step
acc :  0.8069
'''


