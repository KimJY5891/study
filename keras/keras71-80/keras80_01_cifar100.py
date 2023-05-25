
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201,DenseNet121,DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile,NASNetLarge
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB7
from tensorflow.keras.applications import Xception
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input

model_list = [
    VGG19,
    Xception,
    ResNet50,
    ResNet101,
    InceptionV3,
    InceptionResNetV2,
    DenseNet121,
    MobileNetV2,
    NASNetMobile,
    EfficientNetB0
]
# 1. 데이터

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape,x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape,y_test.shape) # (50000, 1) (10000, 1)

for i,v in enumerate(model_list) : 
    # 2. 모델 
    application = v(
        weights='imagenet',
        include_top = False, 
        input_shape = (32,32,3)
    )
    application.trainable = False
    print(len(application.weights)) # 
    print(len(application.trainable_weights)) # 
    
    # 함수 모델 
    input1 = Input(shape=(32, 32, 3))
    app = application(input1)
    gap = GlobalAveragePooling2D()(app)
    output1 = Dense(10,activation='softmax')(gap)
    model = Model(inputs = input1, outputs = output1)
    
    # 3. 컴파일, 훈련 
   
    model.compile(loss='categorical_crossentropy',optimizer = 'adam')
    model.fit(x_train,y_train,verbose=1,epochs=10, batch_size=1)
    
    # 4. 평가 예측
    print('========================================================')
    print('모델명 : ',v().name)
    result =model.evaluate(x_test,y_test) 
    print('result : ',result )
    y_predict=np.argmax(model.predict(x_test),axis=1)
    y_test = np.argmax(y_test,axis=1)
    acc = accuracy_score(y_test,y_predict)
    print('acc : ',acc)
