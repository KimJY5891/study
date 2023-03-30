# 앙상블 모델 두개 앞쳤다는 의미 
# 한개의 모델 보다 여러개를 합쳤을 때 성능ㅇ ㅣ좋아지는 경우가 종종 있었다. 
# 나빠지는 경우가 있긴하다. 
# 99번 별로여도 1번 좋았으면 그걸 해봐야한다. 
# 무조건 좋다는 것은 아니다. 
# 0.01 정도 이지만 엄청 크다
# 시퀀셜에서 가능 할까? - 안된다. 
# 함수형으로 해야한다. 
# 두개의 인풋을 하나의 노드가 받아들이면 되는것이다. 

from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
#1. 데이터
import numpy as np

x1_datasets = np.array([range(100),range(301,401)]) # 예를 들어 삼성, 아모레 주가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)])# 온도 습도 강수량 

print(x1_datasets.shape) # (2,100)
print(x2_datasets.shape) # (3,100)

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T

print(x1.shape) # (100,2)
print(x2.shape) # (100,3)

y = np.array(range(2001,2101)) # 환율 
# 삼성과 아모레가격으로 환율 맞출수 있게 훈련
# 관계가 없기 때문에 정확도가 안좋을 수 있다.
# 

from sklearn.model_selection import train_test_split
'''
x1_train, x1_test, x2_train, x2_test = train_test_split(
    x1, x2, train_size=0.7, random_state=333
)


y_train, y_test = train_test_split(
    y, train_size=0.7, random_state=333
)

랜덤스테이트 값을 똑같이줘야한다. 
'''

x1_train, x1_test, x2_train, x2_test,y_train, y_test  = train_test_split(
    x1, x2,y, train_size=0.7, random_state=333
)
print('x1_train.shape : ', x1_train.shape,'x1_test.shape : ',x1_test.shape) #x1_train.shape :  (70, 2) x1_test.shape :  (30, 2)
print('x2_train.shape : ', x2_train.shape,'x2_test.shape : ',x2_test.shape) #x2_train.shape :  (70, 3) x2_test.shape :  (30, 3)
print('y_train.shape : ', y_train.shape,'y_test.shape : ',y_test.shape) #y_train.shape :  (70,) y_test.shape :  (30,)

#2. 모델 구성 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델 1
input1 = Input(shape=(2,))
dense1 = Dense(200,activation='relu',name='stock1')(input1)
dense2 = Dense(20,activation='relu',name='stock2')(dense1)
dense3 = Dense(30,activation='relu',name='stock3')(dense2)
output1 = Dense(1,activation='relu',name='output1')(dense3)

#2-2. 모델 2
input2 = Input(shape=(3,))
dense11 = Dense(10,activation='relu',name='weather1')(input2)
dense12 = Dense(10,activation='relu',name='weather2')(dense11)
dense13 = Dense(10,activation='relu',name='weather3')(dense12)
dense14 = Dense(10,activation='relu',name='weather4')(dense13)
output2 = Dense(1,activation='relu',name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate
# concatenate : 사슬처럼 엮다는 의미
# 모델 두개를 엮겟다는 의미
# Concatenate -> 클래스
# concatenate -> 함수
# 두 개의 
merge1 = concatenate([output1,output2],name='mg1') # 두 개 이상이라서 리스트 
merge2 = Dense(2,activation='relu',name='mg2')(merge1)
merge3 = Dense(2,activation='relu',name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)
# 이제는 어디서부터 어디까지 모델이다 

model = Model(inputs=[input1,input2], outputs=last_output) 
model.summary()

'''
0인 부분은 연산량 이 


결국 둘 이 합쳐도 거대한 모델이기 대문에 
합치는 모델재료들은 아웃풋 모델 노드는 아무렇게나 주면 된다. 

Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_2 (InputLayer)           [(None, 3)]          0           []

 input_1 (InputLayer)           [(None, 2)]          0           []

 weather1 (Dense)               (None, 10)           40          ['input_2[0][0]']

 stock1 (Dense)                 (None, 10)           30          ['input_1[0][0]']

 weather2 (Dense)               (None, 10)           110         ['weather1[0][0]']

 stock2 (Dense)                 (None, 20)           220         ['stock1[0][0]']

 weather3 (Dense)               (None, 10)           110         ['weather2[0][0]']

 stock3 (Dense)                 (None, 30)           630         ['stock2[0][0]']

 weather4 (Dense)               (None, 10)           110         ['weather3[0][0]']

 output1 (Dense)                (None, 1)            31          ['stock3[0][0]']

 output2 (Dense)                (None, 1)            11          ['weather4[0][0]']

 mg1 (Concatenate)              (None, 2)            0           ['output1[0][0]',
                                                                  'output2[0][0]']

 mg2 (Dense)                    (None, 2)            6           ['mg1[0][0]']

 mg3 (Dense)                    (None, 2)            6           ['mg2[0][0]']

 last (Dense)                   (None, 1)            3           ['mg3[0][0]']

==================================================================================================
Total params: 1,307
Trainable params: 1,307
Non-trainable params: 0
__________________________________________________________________________________________________
'''

# 만들자
#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
import time
start = time.time()
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='loss',
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True
)
mcp=ModelCheckpoint(monitor='loss',mode='min',verbose=1, save_best_only=True,
                    filepath='./_save/MCP/keras52_ensemble1_mcp.hdf5')
model.fit([x1_train,x2_train],y_train,epochs=1000,batch_size=100,callbacks=[es,mcp])
end = time.time()

#4. 평가, 예측
result=model.evaluate([x1_test,x2_test],y_test)
# 값 두개 - result 
# 값 한개 - loss
print('result : ',result)
y_predict = model.predict([x1_test,x2_test])
def rmse(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse_score = rmse(y_test,y_predict)
print('rmse스코어 : ',rmse_score)
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
# 엑스 테스트가 두개 들어가야 한다면 리스트 





