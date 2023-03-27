
#데이콘  따릉이 문제 풀이
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

# 우리가 사용할 수 있도록 바꿔야함 


#1. 데이터

path = "./_data/dacon_wine/"
path_save = "./_save/dacon_wine/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0) #index_col은 인덱스 컬럼이 뭐냐? #인덱스는 데이터가 아니니까 빼야지! 
print("train_csv",train_csv) 
print("train_csv",train_csv.shape) #(5497, 13)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) 
print(test_csv.shape) #(1000, 12)
# 화이트와 레드의 변환 
# 파이썬 맵 펑션도 가능 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의
le.fit(train_csv['type']) #
aaa = le.transform(train_csv['type']) # 0과 1로 변화
print
print(aaa.shape)
print(type(aaa))
#print(np.unique(aaa,return_count=True))
train_csv['type'] = aaa #다시 집어넣기
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])
print(le.transform(['red','white'])) #
print(le.transform(['white','red'])) #
####################################### 결측치 처리 #######################################                                                                                 #
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)
####################################### 결측치 처리 #######################################                                                                                 #

x = train_csv.drop(['quality'],axis=1)
y = train_csv['quality']
print('y의 라벨 값 :',np.unique(y)) #[3 4 5 6 7 8 9]
print("x.shape : ",x.shape)#(5497, 12)
print("y.shape : ",y.shape)#(5497,)
print(y)
#encoder = OneHotEncoder(sparse=False)
# y = y.reshape(1, -1)
# 에러 메시지에서 제시된 대로 array.reshape(-1, 1)을 
# 이용하여 y 데이터를 reshape 해보시기 바랍니다. 
# -1을 이용하면 자동으로 차원을 계산하므로, 아래와 같이 사용할 수 있습니다.
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.85,shuffle=True,random_state=1234
)
scaler = StandardScaler()
# #scaler = MinMaxScaler()
# #scaler = MaxAbsScaler()
# # scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv)

#2. 모델 구성
model =Sequential()
model.add(Dense(200,activation="relu",input_dim=12))
model.add(Dropout(0.2))
model.add(Dense(10,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(40,activation="softmax"))
model.add(Dropout(0.2))
model.add(Dense(20,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(4))
model.add(Dense(10,activation="softmax"))


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',patience=50,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=3000,batch_size=100,verbose=1,validation_split=0.2,callbacks=[es])
# 회귀에서는 매트릭에서 입력해서 볼 수 있다.

# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
print('loss : ',result[0] )
print('acc : ',result[1] )

y_predict=model.predict(x_test)
y_test_acc = np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=-1)
print('y_test_acc : ',y_test_acc)
print('y_predict : ',y_predict )
acc = accuracy_score(y_test_acc,y_predict)
print('acc : ',acc)

y_submit = np.round(model.predict(test_csv))
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['quality'] =y_submit
submission.to_csv(path_save+'submit_0315_01.csv')


#===================================================================================================================
print(train_csv.columns)

print(train_csv.info())

print(train_csv.describe())
#min: 최소값 ,max : 최대값, 50%: 중위값 
print(type(train_csv))
####################################### 결측치 처리 #######################################                                                                                   #
#결측치 처리 1. 제거
#print(train_csv.isnull().sum()) # 결측치가 몇개 있는지 보여줌
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
# dropna : 결측치를 삭제하겠다.
# 변수로 저장하기
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)

#####################train_csv 데이터에서 x와 y를 분리#######################

x = train_csv.drop(['count'],axis=1)#(1328, 9)
#drop : 빼버리겠다.엑시즈 열
#두 개이상 리스트
print("x : ", x)

y = train_csv['count']
print(y) #(1328,)
#####################train_csv 데이터에서 x와 y를 분리#######################

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8517
)
print("x_train.shape01: ",x_train.shape) #(929, 9)
print("y_train.shape01 : ",y_train.shape) #(929, )

#2.모델구성

model =Sequential()
model.add(Dense(200,activation="relu",input_dim=12))
model.add(Dropout(0.2))
model.add(Dense(10,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(40,activation="softmax"))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(10,activation="softmax"))



#3.컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',patience=50,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=3000,batch_size=100,verbose=1,validation_split=0.2,callbacks=[es])

#4. 평가 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
print('loss : ',result[0] )
print('acc : ',result[1] )

y_predict=model.predict(x_test)
y_test_acc = np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=-1)
print('y_test_acc : ',y_test_acc)
print('y_predict : ',y_predict )
acc = accuracy_score(y_test_acc,y_predict)
print('acc : ',acc)

# nan = 데이터가 없다, 원본 데이터 값이 없어서 nan이 나온다. 0은 데이터가 있는 것 
# 결집치 값 처리 첫 번째 걍 0으로 처리
y_submit = np.argmax(model.predict(test_csv))
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['quality'] =y_submit
submission.to_csv(path_save+'submit_0315_06.csv')
