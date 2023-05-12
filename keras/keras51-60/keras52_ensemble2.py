import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense,Input, Conv1D,Flatten
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error 


# 1. 데이터
x1_datasets = np.array([range(100),range(301,401)]) # 예를 들어 삼성, 아모레 주가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)])# 온도 습도 강수량 
x3_datasets = np.array([range(201,301),range(511,611),range(1300,1400)])# 온도 습도 강수량 

print(x1_datasets.shape) # (2,100)
print(x2_datasets.shape) # (3,100)
print(x3_datasets.shape) # (3,100)
x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T

print(x1.shape) # (100,2)
print(x2.shape) # (100,3)
print(x3.shape) # (100,3)

y = np.array(range(2001,2101)) # 환율 

x1_train, x1_test, x2_train, x2_test,  x3_train, x3_test,y_train, y_test  = train_test_split(
    x1, x2,x3,y, train_size=0.7, random_state=333
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
#2-3. 모델 3
input3 = Input(shape=(3,))
dense21= Dense(12,activation='relu',name='x3_01')(input3)
dense22 = Dense(12,activation='relu',name='x3_02')(dense21)
dense23= Dense(12,activation='relu',name='x3_03')(dense22)
output3= Dense(12,activation='relu',name='x3_04')(dense23)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1,output2],name='mg1') # 두 개 이상이라서 리스트 
merge2 = Dense(2,activation='relu',name='mg2')(merge1)
merge3 = Dense(2,activation='relu',name='mg3')(merge2)
last_output = Dense(1,name='last')(merge3)
# 이제는 어디서부터 어디까지 모델이다 

model = Model(inputs=[input1,input2,input3], outputs=last_output) 
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
                    filepath='./_save/MCP/keras52_ensemble1_mcp.hdf5')
model.fit([x1_train,x2_train,x3_train],y_train,epochs=1000,batch_size=100,callbacks=[es,mcp])
end = time.time()


#4. 평가, 예측

loss=model.evaluate([x1_test,x2_test,x3_test],y_test)
print('loss : ',loss)

y_predict = model.predict([x1_test,x2_test,x3_test])
def rmse(y_test,y_predict) :
    return np.sqrt(mean_squared_error(y_test,y_predict))

rmse_score = rmse(y_test,y_predict)
print('rmse스코어 : ',rmse_score)

r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)



'''



'''
