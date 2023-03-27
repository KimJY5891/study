import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.metrics import r2_score, accuracy_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 1. 데이터

path = "./_data/시험/"
path_save = "./_save/samsung/"
datasets_sam = pd.read_csv(path + '삼성전자 주가2.csv',index_col=0,encoding='cp949')
print(datasets_sam.shape) #(3260, 16)
datasets_hyun = pd.read_csv(path + '현대자동차.csv',index_col=0,encoding='cp949')
print(datasets_hyun.shape) #(3140, 16)
print(datasets_sam.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')
print(datasets_hyun.columns)
# Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
#        '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
#       dtype='object')
print(datasets_sam.info())
print(datasets_sam.describe())
print(datasets_hyun.info())
print(datasets_hyun.describe())
print(" datasets_sam['전일비']의 라벨 값 :",np.unique(datasets_sam['전일비'])) #[' ' '▲' '▼']
print(" datasets_hyun['전일비']의 라벨 값 :",np.unique(datasets_hyun['전일비'])) #[' ' '▲' '▼']

# 데이터에 대한 레이블 인코딩 수행
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의
le.fit(datasets_sam['전일비']) #
aaa = le.transform(datasets_sam['전일비']) # 0과 1로 변화
print(aaa.shape)
print(type(aaa))
print(np.unique(aaa,return_count=True))
datasets_sam['전일비'] = aaa 
print(datasets_sam)
datasets_sam['전일비'] = le.transform(datasets_sam['전일비'])
'''

print(le.transform([' ' '▲' '▼'])) #
datasets_sam = datasets_sam.drop(['종가'],axis=1)
datasets_hyun = datasets_hyun.drop(['종가'],axis=1)
datasets_sam = datasets_sam.astype(str).apply(lambda x: x.str.replace(',', '')).astype(np.float64)
datasets_hyun = datasets_hyun.astype(str).apply(lambda x: x.str.replace(',', '')).astype(np.float64)

# 2. 모델 구성
# 3. 컴파일 훈련 
# 4. 평가 예측 
'''
