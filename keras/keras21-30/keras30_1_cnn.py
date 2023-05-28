from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
# Convolution2D = conv2D
# 회사는 보수적+안정적인걸 원한다.


model = Sequential()
#model.add(Dense(10,input_shape=(3,)))
model.add(Conv2D(
        filters=7 ,
        kernel_size=(2,2),
        input_shape=(8,8,1))) #(n,7,7,7)
    #(batch_size,row,colums,channels)
    # batch_size = 이미지 장수
    # channels = 컬러냐 흑백이냐 포토샵 채널 인가?? (3or1)
    #뒤부터 연산
    #input_shape=(5,5,3)
    # 자르는 크기 (2,2)
    #7 = out노드 갯 수
    # 2칸씩 잘라서 4바이4 니까
    # 7장으로 자를거야
model.add(Conv2D(
        filters=7 ,
        kernel_size=(2,2),
        input_shape=(8,8,1)))
model.add(Conv2D( # 위(n,4,4,7)에서 받아서 자르는것이다.
        filters=4, #이름 : 필터(예전:아웃풋)
        kernel_size=(3,3),
        activation='relu'# 웨이트가 마이너스 인것도 있다. 랜덤으로 해서 나왔을 때 마이너스 나온다면 렐루로 0으로 만들어준다. 모든 4차원의 값은
        )) #(n,5,5,4)


model.add(Conv2D(filters=10 ,
                    kernel_size=(2,2)))#(n,4,4,10)
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
