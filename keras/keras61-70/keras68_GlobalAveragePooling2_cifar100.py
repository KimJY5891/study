from tensorflow.keras.datasets import mnist,cifar100
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D

# 난수 고정, 텐서플로우에 대한 난수 고정이고, 가중치는 아님
tf.random.set_seed(337)

#1. 데이터 

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train / 255.
x_test = x_test / 255.
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 100)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 100)

#2. 모델구성 

model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(2,2), padding='valid', activation='relu')) 
model.add(Conv2D(32, 2))  
#model.add(Flatten())
model.add(GlobalAveragePooling2D()) 
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='softmax')) 

model.summary()


#3. 컴파일, 훈련 

import time 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', patience=30, mode='max', 
                   verbose=1, 
                   restore_best_weights=True
                   )
start = time.time()
model.fit(x_train, y_train, epochs=24, batch_size=16, validation_split=0.2, 
          callbacks=[es])
end = time.time()


#4. 평가, 예측

results = model.evaluate(x_test, y_test)
print('loss:', results[0]) 
print('acc:', results[1]) 
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1) 
y_test = np.argmax(y_test,axis=1)
acc = accuracy_score(y_test, y_pred)
print('acc:', acc)
print('걸린 시간 : ', round(end-start,2),'초')


'''
flatten
loss: 3.5231475830078125
acc: 0.21660000085830688
acc: 0.2166
걸린 시간 :  1101.11 초
\nflatten\n\n\n\nGlobalAveragePooling2D\n\n\n\n


GlobalAveragePooling2D
loss: 3.185181140899658
acc: 0.2146
걸린 시간 :  1103.36 초

'''
