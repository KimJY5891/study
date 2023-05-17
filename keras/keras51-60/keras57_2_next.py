import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
print(x_train.shape) 
print(x_train[0].shape) 
print(x_train[1].shape) 
print(x_train[0][0].shape) 

print(np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1).shape) # (100, 28, 28, 1)

print(np.zeros(augment_size)) 
print(len(np.zeros(augment_size))) 
print(np.zeros(augment_size).shape) 
x_data= train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),# x 데이터 
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=True
).next()
#########################. next()미사용 #########################
print('x_data : ', x_data) # x와 y가 합쳐진 데이터 출력
# 첫 번쩨인 x_data[0]의 값이 나옴 
print(type(x_data)) # <class 'tuple>
print(x_data[0]) # x데이터
print(x_data[1]) # x데이터
print(x_data[0].shape,x_data[1].shape) 
print(type(x_data[0]))
# 튜플 (x,y) - 이터레이터
# 둘 다 넘파이 
#########################. next()미사용 ######1###################
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49) : 
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i],cmap='gray') #.next() 미사용
    #plt.imshow(x_data[0][0][i],cmap='gray') #.next() 미사용 (x_data[0][0][i] -> 배치가 있어서 이런 식으로 작성했다.  [배치 ][?][ ?]
    #  x_data[0][i] - 통배치는 앞에 하나만 줄어들면 된다.  
plt.show()

'''
출처 : gpt
next()는 파이썬 내장 함수로, 이터레이터(Iterator) 객체에서 다음 요소를 반환하는 역할을 합니다. 
이터레이터는 값을 차례대로 반환하는 객체로, 리스트, 튜플, 문자열 등과 같은 시퀀스 데이터를 순회할 수 있도록 도와줍니다.

next() 함수는 다음과 같은 방식으로 사용됩니다:

python
Copy code
next(iterator, default)
iterator: 값을 반환할 이터레이터 객체입니다.
default (옵션): 이터레이터가 모든 요소를 반환한 후에 호출되었을 때 반환할 기본 값입니다.
이 옵션이 없을 경우, 이터레이터가 모든 요소를 반환한 후 StopIteration 예외가 발생합니다.
next() 함수는 이터레이터의 다음 요소를 반환합니다.
이 때, 이터레이터는 내부적으로 상태를 유지하고 있으며, next() 함수가 호출될 때마다 다음 요소를 반환합니다.
이터레이터가 더 이상 반환할 요소가 없다면, StopIteration 예외가 발생합니다.
이 때, default 값이 지정되어 있다면 default 값을 반환하게 됩니다.

아래는 next() 함수의 예제입니다:

python
Copy code
my_list = [1, 2, 3, 4, 5]
my_iter = iter(my_list)  # 이터레이터 생성

print(next(my_iter))  # 1
print(next(my_iter))  # 2
print(next(my_iter))  # 3

# 모든 요소를 반환한 후에 StopIteration 예외 발생
print(next(my_iter))  # StopIteration 예외 발생

# 기본 값으로 'End'를 반환
print(next(my_iter, 'End'))  # 'End'
이 예제에서는 my_list라는 리스트를 이터레이터 객체로 변환한 후
, next() 함수를 사용하여 다음 요소를 차례대로 반환하고 있습니다.
마지막에는 default 값을 지정하여, 이터레이터가 모든 요소를 반환한 후에는
'End' 값을 반환하도록 하고 있습니다.'
'''


