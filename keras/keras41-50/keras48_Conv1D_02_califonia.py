from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,LSTM,Conv1D,Flatten


# 1. 데이터
datasets = fetch_california_housing()
x=datasets.data
y=datasets.target
print(x.shape,y.shape) #(20640, 8) (20640,)
x=x.reshape(20640,8,1)
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715,shuffle=True
)


#2. 모델


input01 = Input(shape=(8,1))
Conv1D01 = Conv1D(10, 2, padding='same')(input01)
Conv1D02 = Conv1D(10, 2)(Conv1D01)
Flatten01 = Flatten()(Conv1D02)
Dense01 = Dense(12)(Flatten01)
Dense02 = Dense(12)(Dense01)
output01 = Dense(1)(Dense02)
model = Model(inputs=input01, outputs=output01)




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
mcp= ModelCheckpoint(monitor='loss',mode='min',verbose=1,save_best_only=True,
                     filepath='./_save/MCP/keras48_Conv1D_02_califonia_mcp.hdf5'
                     )
model.fit(x_train,y_train,epochs=1000,batch_size=100,callbacks=[es,mcp])
end = time.time()


#4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss: ',loss)
y_predict = model.predict(x_test)


r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)
