# 드랍 아웃에 좋은 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(type(x))
x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8,random_state=333    
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train),np.max(x_train))#0.0 1.0
print(np.min(x_test),np.max(x_test))#0.0 1.0
"""
#2. 모델
# 시퀀셜 모델
model =Sequential()
model.add(Dense(30,input_shape=(13,)))
model.add(Dropout(0.3))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
# 모델.이벨류 에이트에 들어가는 연산값에는 드랍아웃은 들어가지 않는다. 
# 가중치가 생성됐을 때

#함수형 모델 
# input01 = Input(shape=(13,)) #input01 = 레이어 이름 
# demse01 = Dense(30)(input01)
# demse02 = Dense(20)(demse01)
# demse03 = Dense(10)(demse02)
# output01 = Dense(1)(demse03)

#함수형 모델 
# model = Model(inputs=input01,outputs=output01)

#컴파일
model.compile(loss='mse',optimizer='adam')
import datetime
date= datetime.datetime.now()
print("date :",date)
date= date.strftime('%m%d_%H%M')#스트링퍼 타임 문자로 바꿔야 
print("date :",date)
#date : 2023-03-14 11:15:53.259423
filepath = './_save/MCP/keras27/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
                   verbose=1, #디폴트 0
                   restore_best_weights=True,
                   )
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True,  
                    filepath=''.join((filepath,'k27_',date,'_',filename))# 합칠거야 
                    )
model.fit(x_train,y_train,
          epochs=1220,batch_size=72,
          callbacks=[es,],#mcp],
          validation_split=0.2
          )
#model = load_model('./_save/MCP/keras27_ModelCheckPoint.hdf5')
# model.save('./_save/MCP/keras27_3_save_model.hdf5')
# 두개의 가중치 저장됌

#4.평가,예측
print('====================1. 기본출력 =================')
loss=model.evaluate(x_test,y_test,verbose=0)
print('loss: ',loss)
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
"""
