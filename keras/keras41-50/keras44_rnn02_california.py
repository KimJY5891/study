import time 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input,LSTM ,SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# 1. 데이터

datasets=fetch_california_housing()
x=datasets.data # (20640, 8)
y=datasets.target # (20640,)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715,shuffle=True
)

scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#2. 모델

model =Sequential()
model.add(SimpleRNN(128, input_shape=(x_train.shape[0],1)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련 
start = time.time()
model.compile(optimizer = 'adam',loss='mse')
es = EarlyStopping(monitor='val_loss',patience=45,mode='min',
                   verbose=1,restore_best_weights=True)
model.fit(x_train,y_train, epochs=128,batch_size=256,
          validation_split=0.1, verbose=1, callbacks=[es])
end = time.time()
print(f'걸린시간 : {end-start } 초 ')

#4 평가, 예측

loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
r2 = r2_score(y_test,y_pred)
print("r2 : ", r2)

# r2 :  0.741675518224558
