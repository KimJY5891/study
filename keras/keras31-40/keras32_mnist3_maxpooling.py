import numpy as np
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler #전처리
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data() # 데이터를 넣어줌

scaler = MinMaxScaler()
x_train = x_train.reshape(-1,1)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)
print("x_train.shape: ",x_train.shape)
print("y_train.shape : ",y_train.shape)
#3차원데이터
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1) # 데이터의 구조만 바귀는 것 순서와 값이 바뀌는 게 아님

y_train= to_categorical(y_train)
y_test= to_categorical(y_test)
print(y_train)
print(y_train.shape)

#2. 모델
model =Sequential()
model.add(Conv2D(10,(2,2),padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D()) # conv2D(cnn)과 다르게 중첩되지 않는다. 계산을 두 번하면 건너뛰게 되어있다.
model.add(Conv2D(filter=64,kernel_size=(2,2),
                  padding='valid', activation='relu'))
model.add(Conv2D(32,2)) #2 = (2,2) 줄임버전
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.summary()


'''
gpt설명
MaxPooling2D는 합성곱 신경망(Convolutional Neural Network, CNN)에서 주로 사용되는 풀링 연산입니다.
이미지 처리 작업에 자주 쓰이며, 입력된 특징 맵의 공간적인 크기(너비와 높이)를 줄이면서 가장 중요한 정보를 유지합니다.

MaxPooling2D 연산은 입력 이미지나 특징 맵을 일련의 겹치지 않는 직사각형 영역(보통 풀링 윈도우 또는 커널이라고 함)으로 나누고,
각 영역에서 최대값을 추출합니다. 풀링 윈도우의 크기와 스트라이드(윈도우가 이동하는 양)는 지정된 매개변수에 의해 결정됩니다.

MaxPooling2D의 작동 방식은 다음과 같습니다:

입력 이미지나 특징 맵은 지정된 풀링 윈도우 크기에 따라 겹치지 않는 영역으로 나누어집니다.
각 영역에서 최대값이 추출됩니다.
최대값들은 새로운 크기로 줄어든 출력 특징 맵으로 정렬되어 저장됩니다.
MaxPooling2D의 주요 목적은 입력의 공간적인 크기를 줄이는 것으로, 네트워크의 매개변수와 계산량을 감소시킵니다.
최대값만을 유지함으로써, 중요한 특징을 보존하면서 덜 중요한 세부 사항을 제거합니다.
이는 네트워크가 작은 공간적인 이동에 더 강인해지도록 하여, 변화에 더 강인한 특성을 추출하는 데 도움이 됩니다.

MaxPooling2D는 일반적으로 CNN 아키텍처에서 합성곱 층 이후에 적용됩니다.
이는 점진적으로 특징 맵의 공간적인 크기를 줄여줌으로써 더 높은 수준의 특징을 추출합니다. 
MaxPooling2D는 여러 번 적용될 수 있으며, 이에 따라 특징 맵은 계층적으로 줄어들게 됩니다.
'''


