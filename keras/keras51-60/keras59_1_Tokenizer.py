import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛잇는 밥을 엄청 마구 마구 마구 먹었다.'
# 띄어 쓰기잖아 -> 음절
# 컴퓨터가 텍스트를 인식하기 위해서는 수치화 해줘야함

token = Tokenizer()
# 전처리 
token.fit_on_texts(
    [text]
) # 문장이 여러 문장이 있을 수 잇다. 그럴 경우 리스트
print(token.word_index) 
# 키밸류 형태
# {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛잇는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# 중복된 건 또 작성은 안햇다. 
# word_index 글을 인덱싱 
# 가장 많은 애가 가장 앞으로 가삳. 
print(token.word_counts)
# 키밸류 형태
# OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맛잇는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])
# 단어가 나온 형태 

x = token.texts_to_sequences(
    [text]
) # 여러개있을 수 도 잇기 때문에 리스트 형태로 작성한다.
# 숫자로 바꿔주는 작업
# 결과를 변수(메모리)에 넣어준다. 
print(x) 
# [[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]]
print(np.array(x).shape) # (1, 11)
# 그냥 연산하면 숫자에 따라서 가치가 높다고 생각하게 된다. 
# 위에 수치는 가치가 다른것이 아니라 평등 관게이다. 
# 가치가 같으니까 - 원핫 인코딩을 해줘야한다. 
######################## 1. to_categorical ########################
from tensorflow.keras.utils import to_categorical
# x  = to_categorical(x)
# print(x)
# [[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
# print(x.shape) #(1, 11, 9)
# (1, 11, 8)(1~8)이 아닌 이유 - to 카테고리 컬은 무조건 0부터 시작함
# 그래서 필요 없는 0 있음
# 0을 지워주고 리쉐이프 해줘야함
 
######################## 2. get_dummies ########################
import pandas as pd  
# x = pd.get_dummies(x) # 1차원으로 받아들여야한다. 
# print(x)
# Traceback (most recent call last):
#   File "c:\study\keras\keras59_Tokenizer.py", line 50, in <module>
#     print(x.shape) #(1, 11, 9)
# AttributeError: 'list' object has no attribute 'shape'
# 겟더미에서 리스트 형태를 받아들일 수 없다. 
# 1. 넘파이
# x = np.array(x)
# x = 플랫튼과 동일 x.ravel()
# x = x.flatten()
# x = pd.get_dummies(x)
# print(x.shape)
# x = np.array(x)
# x = x.reshape(11,)
# x = pd.get_dummies(x)
# print(x.shape)
# 2. 왜 리스트를 받아들이지 못할까?
# sklearn.preprocessing.OneHotEncoder를 사용하여 
# 변환된 결과는 numpy.array이기 때문에
# 이를 데이터프레임으로 변환하는 과정이 필요하다.


######################## 3. 사이킷런 onehot ########################
from sklearn.preprocessing import OneHotEncoder # 2차원으로 받아들여야한다. 
# 알아서 만들자
ohe = OneHotEncoder(sparse=False)
x = np.array(x)
x = x.reshape(-1,1)
print(x)
print(x.shape) # 가로로 값이 있느넥 아니라 # 변환 할 때 세로로 변환 시켜서 0010000 이런 형태로 바궈줘야한다. 
x = ohe.fit_transform(x)
print(x)
print(x.shape)
# fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다

# print(x)
# ohe = OneHotEncoder(sparse=False)
# x = np.array(x)
# print('x1 : ',x.shape)
# x = x.ravel() -> 안됌 
# print(x.shape)
# x = ohe.fit_transform(x)
# print(x)
# print(x.shape)(11, 8)
