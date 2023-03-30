from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 
# 1. 데이터
import numpy as np

x1_datasets = np.array([range(100),range(301,401)]) # 예를 들어 삼성, 아모레 주가
# x2_datasets = np.array([range(101,201),range(411,511),range(150,250)])# 온도 습도 강수량 
# x3_datasets = np.array([range(201,301),range(511,611),range(1300,1400)])# 온도 습도 강수량 

print(x1_datasets.shape) # (2,100)
# print(x2_datasets.shape) # (3,100)
# print(x3_datasets.shape) # (3,100)
x1 = np.transpose(x1_datasets)
# x2 = x2_datasets.T
# x3 = x3_datasets.T
# 엑스를 하나로 만들어서 수정 
print(x1.shape) # (100,2)
# print(x2.shape) # (100,3)
# print(x3.shape) # (100,3)

y1 = np.array(range(2001,2101)) #환율
y2 = np.array(range(1001,1101)) #금리

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test,y2_train, y2_test  = train_test_split(
    x1,y1,y2, train_size=0.7, random_state=333
)#역슬래시 하면 두줄이 한 줄이다. 
print('x1_train.shape : ', x1_train.shape,'x1_test.shape : ',x1_test.shape) #x1_train.shape :  (70, 2) x1_test.shape :  (30, 2)
# print('x2_train.shape : ', x2_train.shape,'x2_test.shape : ',x2_test.shape) #x2_train.shape :  (70, 3) x2_test.shape :  (30, 3)
# print('x2_train.shape : ', x3_train.shape,'x2_test.shape : ',x3_test.shape) #x2_train.shape :  (70, 3) x2_test.shape :  (30, 3)
print('y_train.shape : ', y1_train.shape,'y_test.shape : ',y1_test.shape) #y_train.shape :  (70,) y_test.shape :  (30,)
print('y_train.shape : ', y2_train.shape,'y_test.shape : ',y2_test.shape) #y_train.shape :  (70,) y_test.shape :  (30,)

# 아래 코드도 앙상블 모델인가요??
#2. 모델 구성 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. x모델 1
input1 = Input(shape=(2,))
dense1 = Dense(200,activation='relu',name='stock1')(input1)
dense2 = Dense(20,activation='relu',name='stock2')(dense1)
dense3 = Dense(30,activation='relu',name='stock3')(dense2)
output1 = Dense(1,activation='relu',name='output1')(dense3)

#2-2. 분기 모델 1
bungi1= Dense(10,activation='relu',name='bg1')(output1)
bungi2= Dense(10,activation='relu',name='bg2')(bungi1)
last_output1= Dense(1,activation='relu',name='last_output1')(bungi2)

#2-3. 분기 모델 1
bungi11= Dense(10,activation='relu',name='bg11')(output1)
bungi12= Dense(10,activation='relu',name='bg12')(bungi11)
last_output2= Dense(1,activation='relu',name='last_output2')(bungi12)

model = Model(inputs=[input1], outputs=[last_output1,last_output2]) 
model.summary()

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
                    filepath='./_save/MCP/keras52_ensemble2_mcp.hdf5')
model.fit([x1_train],[y1_train,y2_train],epochs=1000,batch_size=100,callbacks=[es,mcp])
end = time.time()


#4. 평가, 예측

loss=model.evaluate([x1_test],[y1_test,y2_test])
print('loss : ',loss)

y_predict = model.predict([x1_test])
y_predict = np.array(y_predict)
def rmse(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test,y_predict))
print(y_predict)
print(y_predict.shape)

rmse_score01 = rmse(y1_test,y_predict[0])
rmse_score02 = rmse(y2_test,y_predict[1])
rmse_score = (rmse_score01+rmse_score01)/2
# print('rmse스코어 : ',rmse_score)
print('rmse_score01스코어 : ',rmse_score01)
print(' rmse_score02 스코어 : ',rmse_score02)
print(' rmse 스코어 : ',rmse_score)

r2_01=r2_score(y1_test,y_predict[0])
r2_02=r2_score(y2_test,y_predict[1])
r2=(r2_01+r2_02)/2
print('r2_01스코어 : ',r2_01)
print('r2_02스코어 : ',r2_02)
print('r2스코어 : ',r2)

# 화요일 종가 맞추기 30점 
# 수요일 싯가 맞추기 30점 
