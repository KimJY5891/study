import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 

#1. 데이터 
traindatagen=ImageDataGenerator( # 증폭옵션
    rescale=1./255, #스케이링 하겠다. 이미지 값 : 0 ~ 255 - > 0~1 사이로 정규화하겠다. 
    # 정규화 민맥스 스케일러로 전처리 하겠다. 노멀라이제이션
    # 부동소수점으로 연산을 해라 
   
    horizontal_flip=True, # 아무튼 반전 
    vertical_flip=True ,# 아무튼 반전 
    width_shift_range=0.1, # 좌우로 움직이는것을 10프로 만큼의 사진을 좌우로 이동할 수 있는?
    height_shift_range=0.1, # 
    rotation_range=5, # 돌리는것 
    zoom_range=1.2, # 약 20프로 확대 하겠다.
    shear_range= 0.7,#찌그렇뜨리는 것 
    fill_mode='nearest' # 빈 자리를 0 말고  원래 있는 값의 근처갑으로 채워라 
# 백프로 다하느 ㄴ것이 아니라 몇개씩 골라서 하는것이다. 
# 머신이 사람이 누군인지 인식하려고 할때 옆으로 뒤집어있어서도 그 사람이 누군인지 알수 있기 대문에 데이터로 사용할 수 있다. 
# 숫자 6하고 9는 상하 반전 사용하면 안된다.  (why?)
)

testdatagen=ImageDataGenerator(
    rescale=1./255,
    # 통상적으로 테스트데이터는 증폭하지 않는다. 
    # 평가데이터를 증폭한다는 것은 조작하는 것이다.  
    # 왜 이렇게 하면안된다. 
    # horizontal_flip=True,
    # vertical_flip=True , 
    # width_shift_range=0.1, 
    # height_shift_range=0.1, 
    # rotation_range=5, 
    # zoom_range=1.2,
    # shear_range= 0.7,
)
xy_train = traindatagen.flow_from_directory(
    'c:/study_data/_data/brain/train/',
    #'d:/study_data/_data/brain/train/',
    target_size= (200,200), 
    batch_size=5,
    class_mode='binary',# 수치화해서 만들어 준다. 0과 1로 변한다.  int형
    color_mode='grayscale', # 컬러도 있음
    shuffle=True
) #Found 160 images belonging to 2 classes. 160 이미지 갯수 데이터는 그대로 증폭으 ㄴ안됌 
# directory = folder 
# 분류 되어있는 폴더를 상위까지 지정해주면 된다.  (이유까먹음 알아내야함 )
# 이미지의 가로 세로 크기가 다를 수 있다. 
# 라벨링하다가 크기가 달라질 수 도 잇다. 
# 같은 크기가 아닌 즉 같은 쉐이프가 아니면 훈련이 안됌 
# 그래서 크기를 맞춰줘야함 
# 타켓 사이즈란 이 이미지 크기로 확대 축소하겟다는 것을 해야함 
# 잘못 설정시 확 눌릴 수 있기 때문에 잘 조절해줘야한다. 
# y = (160,)

xy_test = testdatagen.flow_from_directory(
    'c:/study_data/_data/brain/test/',
    #'d:/study_data/_data/brain/test/',
    target_size= (200,200), 
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',  # (200,200,1) (가로,세로,색상)
    shuffle=True
) # Found 120 images belonging to 2 classes. => 120 이미지 갯수 
# 판다스의 밸류 카운트에서 확인했을 때 0과1의 클라스 
# 트레인 폴더에 라벨을 만들위한 두개의 폴더 
# 테스트 폴더에 검증을 위한 두개의 폴더  정상이냐 비정상이냐 
# 이런식으로 폴더 구조를 ㅁ나들어져있음 
# 그것을 이미지 제너레이션으로 만들어엿더니 

print(xy_train) #<keras.preprocessing.image.DirectoryIterator object at 0x00000258EA4C5F10>
# <keras.preprocessing.image.DirectoryIterator object at 0x000001E08BE58F70> 
# Iterator = 반복자 
# 0x000001E08BE58F70 -> 메모리 주소값 
print(xy_test) #<keras.preprocessing.image.DirectoryIterator object at 0x00000258F9678370>
print(xy_train[0])
#  [0.04313726]]]], dtype=float32), array([1., 0., 0., 0., 0.], dtype=float32))
# print(np.array(xy_train[0]).shape)
# y = array([1., 0., 0., 0., 0.] 
# shape는 넘파이에서만 먹힌다.
print(len(xy_train))       #32 배치 크기 순서대로 잘려잇다. 
print(len(xy_train[0]))    #2 xy가 들어가있다. 
print(xy_train[0][0])    #x 5개
print(xy_train[0][1])    #y [1. 0. 0. 0. 1.]
print(xy_train[0][0].shape) #(5, 200, 200, 1)
print(xy_train[0][1].shape) #(5,)
# (배치사이즈, 가로, 세로, 컬러갯수(채널))
# [0]번째에 xy가 있고, 배치만큼 들어가 있다. 
# 데이터 나누기 5 이기에 [31]번째까지 잇다. (총 32이게 )
# 즉 엑스와이 트레인은 0부터 3번까지 잇다. 
# [모든]번째의 [0]번째는 x
# [모든]번째의 [1]번째는 y
# 

print('==========================================================================')
print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # xy_train[0]은 1번째 배치 | <class 'tuple'> - 못바꾼다. 
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>



#2. 모델 구성 
#3. 
