# save_model
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler #전처리

datasets = load_boston()
x = datasets.data
y = datasets.target
print(type(x))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8,random_state=333    
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train),np.max(x_train))#0.0 1.0
print(np.min(x_test),np.max(x_test))#0.0 1.0

#2. 모델
#시퀀셜 모델
# model =Sequential()
# model.add(Dense(30,input_shape=(13,)))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

#함수형 모델 
input01 = Input(shape=(13,)) # input01 = 레이어 이름 
demse01 = Dense(30)(input01)
demse02 = Dense(20)(demse01)
demse03 = Dense(10)(demse02)
output01 = Dense(1)(demse03)

model = Model(inputs=input01,outputs=output01)


# 3. 컴파일,
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
                   verbose=1, #디폴트 0
                   restore_best_weights=True,)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,  
                    filepath='c:/study/_save/MCP/keras27_3_MCP.hdf5')
model.fit(x_train,y_train,
          epochs=1220,batch_size=72,callbacks=[es,
                                               # mcp
        ],validation_split=0.2)
model.save('c:/study/_save/MCP/keras27_3_save_model.h5')
# 두 개의 가중치 저장됌

# 4. 평가, 예측
print('==================== 1. 기본출력 =================')
loss=model.evaluate(x_test, y_test, verbose=0)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)

print('==================== 2. load_model출력 =================')
model02 = load_model('c:/study/_save/MCP/keras27_3_save_model.h5')
loss=model02.evaluate(x_test, y_test, verbose=0)
print('loss: ',loss)
y_predict = model02.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)

print('==================== 3. MCP출력 =================')
model03 = load_model('c:/study/_save/MCP/keras27_3_save_model.hdf5')
loss=model03.evaluate(x_test,y_test,verbose=0)
print('loss: ',loss)
y_predict = model03.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)

"""
# 베스트가중치 : 
==================== 1. 기본출력 =================
loss:  23.96661949157715
r2스코어 :  0.7556395936458179
==================== 2. load_model출력 =================
loss:  23.96661949157715
r2스코어 :  0.7556395936458179
==================== 3. MCP출력 =================
loss:  24.05989646911621
r2스코어 :  0.7546885525793313
""
