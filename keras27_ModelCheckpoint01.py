from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

datasets = load_boston()
x = datasets.data
y = datasets.target

print(type(x))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8,random_state=333    
)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
#위에 두개를 합친것
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)
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
input01 = Input(shape=(13,)) #input01 = 레이어 이름 
demse01 = Dense(30)(input01)
demse02 = Dense(20)(demse01)
demse03 = Dense(10)(demse02)
output01 = Dense(1)(demse03)

#함수형 모델 
model = Model(inputs=input01,outputs=output01)

#컴파일
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
                   verbose=1, #디폴트 0
                   restore_best_weights=True,
                   validation_split=0.2
                   )
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',
                    verbose=1,save_best_only=True, # 가장 낮은 마지막 지점 
                    filepath='./_save/MCP/keras27_ModelCheckPoint.hdf5'# 가중치 저장
                    )#가장 낮은 지점에서 가중치 저장
#저장하는 것이 목적 
model.fit(x_train,y_train,
          epochs=1220,batch_size=72,callbacks=[es,mcp]
          )

#4.평가,예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
