# 지금까지는 2차원
# 가로 세로 컬러 장수
# INPUT.SHAPE -> 들어가는 인풋에 대한 모양 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScalar #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
# 일요일에 정규분포 같은 impot 거 정리해서 보내기 
# Load the dataset
# 메일 : 서너줄 정도 
# 수식은 쓰지 말자 
datasets = load_boston()
x = datasets.data
y = datasets.target

# Print the type of x
print(type(x))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8,random_state=333    
)
# 모르는거 한 번 씩 돌려보기 
scaler = StandardScalar() #X변환 
#scaler = MinMaxScaler() #X변환 
scaler.fit(x_train)  # 변환 기준
#x_train 변환 범위에 맞춰서 변환해야한다. 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
print(np.min(x_train),np.max(x_train))#0.0 1.0
print(np.min(x_test),np.max(x_test))#0.0 1.0

#2. 모델

model =Sequential()
model.add(Dense(1,input_shape=(13,))) #input_shape= 3차원 가능 스칼라 13개 벡터 1개 
# 3차원이면 (시계열 데이터)
# (1,1,1)  >>> input_shape=(1,1)) 
# 맨 앞에 있는건 데이터의 갯수
# 4차원 (이미지 데이터)
# (4,3,2,1) >>> input_shape=(3,2,1)
# 모두 input_shape로 활용
# model.add(Dense(1,input_dim=13)) #13은 행무시 2차원 데이터에서는 행무시, input_dim

#3. 컴파일, 훈련 
model.compile(loss='mse',optimize='adam')

#4. 평가, 예측

# 우리는 수치로 예측 할 것인가
# 모델 두 개로 돌리기! 
# 두개의 데이터를 가지고 삼성전자으 ㅣ 주가르 맞출 것이다. 
# 강도 습도 데이터 모아서 모델에 합쳐서 삼성전자의 주가를 맞출 것이다.
# 시퀀셜로 가능 할까? 
# 함수 = 재사용
# 
 
