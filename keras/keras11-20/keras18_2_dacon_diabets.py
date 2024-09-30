import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

#1. 데이터

path = "./_data/dacon_diabets/"
path_save = "./_save/dacon_diabets/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
####################################### 결측치 처리 #######################################                                                                                 #
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.info()) # 함수를 이용하여 데이터 df의 정보를 출력함 
print(train_csv.shape)
####################################### 결측치 처리 #######################################                                                                                 #
x = train_csv.drop(['Outcome'],axis=1)#
y = train_csv['Outcome']
print("x.shape : ",x.shape)#(652, 8)
print("y.shape : ",y.shape)#(652, )
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=1234
)
#2. 모델 구성
model =Sequential()
model.add(Dense(200,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(100,activation="relu"))
model.add(Dense(200,activation="relu"))
model.add(Dense(400,activation="sigmoid"))
model.add(Dense(800,activation="relu"))
model.add(Dense(80,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="relu"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="linear"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))
#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',patience=160,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=3000,batch_size=1000,verbose=1,validation_split=0.2,callbacks=[es])
# 회귀에서는 매트릭에서 입력해서 볼 수 있다. 

# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

y_submit = np.round(model.predict(test_csv))
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['Outcome'] =y_submit
submission.to_csv(path_save+'submit_0311_19.csv')
#456
"""#2. 모델 구성
model =Sequential()
model.add(Dense(10,activation="relu",input_dim=8))
model.add(Dense(8,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(80,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="linear"))
model.add(Dense(4,activation="linear"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=700,batch_size=800,verbose=1,validation_split=0.2,)
# 회귀에서는 매트릭에서 입력해서 볼 수 있다. 
#mse  wl
# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)

y_submit = np.round(model.predict(test_csv))
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['Outcome'] =y_submit
submission.to_csv(path_save+'submit_0309_1320.csv')
"""
"""
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(8,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(400,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=400,verbose=1,validation_split=0.2,callbacks=[es])
Restoring model weights from the end of the best epoch.
Epoch 00110: early stopping
3/3 [==============================] - 0s 997us/step - loss: 0.6832 - accuracy: 0.6061 - mse: 0.2437
result :  [0.6832188963890076, 0.6060606241226196, 0.24365873634815216]
acc :  0.6060606060606061

"""
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=8715,
)

#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(8,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="linear"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=100,verbose=1,validation_split=0.2,callbacks=[es])
Restoring model weights from the end of the best epoch.
Epoch 00232: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 0.6688 - accuracy: 0.5909 - mse: 0.2318
result :  [0.6688427329063416, 0.5909090638160706, 0.23180462419986725]
acc :  0.5909090909090909
"""

"""
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(8,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Restoring model weights from the end of the best epoch.
Epoch 00139: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 0.6447 - accuracy: 0.6061 - mse: 0.2301
result :  [0.6446599364280701, 0.6060606241226196, 0.2301253080368042]
acc :  0.6060606060606061
submission.to_csv(path_save+'submit_0309_1330.csv')
"""

"""
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(200,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Restoring model weights from the end of the best epoch.
Epoch 00166: early stopping
3/3 [==============================] - 0s 2ms/step - loss: 0.7075 - accuracy: 0.5606 - mse: 0.2565
result :  [0.7075445652008057, 0.560606062412262, 0.2564668655395508]
acc :  0.5606060606060606
"""
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=8715,
)

#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="sigmoid"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=120,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Restoring model weights from the end of the best epoch.
Epoch 00126: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 0.6643 - accuracy: 0.5606 - mse: 0.2372
result :  [0.664304792881012, 0.560606062412262, 0.23716986179351807]
acc :  0.5606060606060606
"""
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=8715,
)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy',patience=120,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Epoch 00121: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 0.6909 - accuracy: 0.5606 - mse: 0.2489
result :  [0.6908721327781677, 0.560606062412262, 0.2488626390695572]
acc :  0.5606060606060606
submission.to_csv(path_save+'submit_0309_1418.csv')
"""
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=8715,
)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(400,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy',patience=120,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Restoring model weights from the end of the best epoch.
Epoch 00523: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 2.3089 - accuracy: 0.6970 - mse: 0.2810
result :  [2.3088903427124023, 0.6969696879386902, 0.2810487151145935]
acc :  0.696969696969697
 val_mse: 0.2705
3/3 [==============================] - 0s 989us/step - loss: 1.9824 - accuracy: 0.6667 - mse: 0.3074
result :  [1.982370376586914, 0.6666666865348816, 0.30743616819381714]
acc :  0.6666666666666666
"""
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=6546,
)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy',patience=120,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
"""
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=6546,
)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy',patience=20,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Epoch 00031: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 0.6499 - accuracy: 0.5758 - mse: 0.2318
result :  [0.6498667597770691, 0.5757575631141663, 0.2317691445350647]
acc :  0.5757575757575758
""""""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=6546,
)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy',patience=20,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Restoring model weights from the end of the best epoch.
Epoch 00635: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 1.8787 - accuracy: 0.6667 - mse: 0.2843
result :  [1.8787319660186768, 0.6666666865348816, 0.2843185365200043]
acc :  0.6666666666666666
"""
"""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=6546,
)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy',patience=20,mode='max',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1,validation_split=0.2,callbacks=[es])
Epoch 00031: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 0.6499 - accuracy: 0.5758 - mse: 0.2318
result :  [0.6498667597770691, 0.5757575631141663, 0.2317691445350647]
acc :  0.5757575757575758
""""""
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=6546,
)
#2. 모델 구성
model =Sequential()
model.add(Dense(100,activation="relu",input_dim=8))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(800,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import Ea
