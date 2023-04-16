from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
# Convolution2D = conv2D

#2. 모델 구성
model = Sequential()
#model.add(Dense(10,input_shape=(3,)))
model.add(Conv2D(7 ,(2,2),
    padding='valid',
    strides=1,
    input_shape=(9,9,1))) #(n,7,7,7)
model.add(Conv2D(
    filters=7 ,
    padding='same',
    # 디폴드 valid    
    kernel_size=(2,2)))
model.add(Conv2D( 
    filters=4,
    kernel_size=(3,3),
    activation='relu'
    )) #(n,5,5,4)
model.add(Conv2D(filters=10 ,
                 kernel_size=(2,2)))#(n,4,4,10)
model.summary()
"""

# conv2D 레이어 갯수는 아무렇게나 상관없다.


# 양은 줄었지만 특성은 더 높아졋다. 서로 구별이 가능해졋다.
# 그림 3개중에서 찾아야하는데
# 레이어의 노드가 40개가 됐다.
# 입체적인 값을 쫙 펴서 계산해준다. 노드는 40게
model.add(Flatten())#(n,4x4x10)
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
#Dense =2 차원 모델만 받아들일ㅇ수 잇다.
#Desne =  wx+b = y
# 찾아야할 것
# 필터,input
model.summary()    
# 파라미터가 왜 이런 값이 나오는지
"""
"""
공식 : 채널 x kernel_size x 필터 + 필터


model.add(Conv2D(
    filters=7 ,
    kernel_size=(2,2),
    input_shape=(8,8,1))) #(n,7,7,7)


1x(2x2)x7 + 1 =35


model.add(Conv2D(
    filters=4,
    kernel_size=(3,3),
    activation='relu
    ))  ##(n,5,5,4)
7x(3x3)x4+4 = 256


model.add(Conv2D(filters=10 ,
                 kernel_size=(2,2))
4x2x2x10+10=170
"""


"""
 스트라이트  디폴트 : 1 
 (보폭)
 그림이 한칸씩 가는거 
 커널 사이즈 보다 많이 주지는 않을 것 
 
"""
