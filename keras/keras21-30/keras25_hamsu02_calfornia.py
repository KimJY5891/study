from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test,y_train,y_test = train_test_split(
    x,y, test_size=0.2,shuffle=True,random_state=337
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

# 시퀀셜 모델 
# model = Sequential()
# model.add(Dense(5, input_dim = 8))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(1))

# 함수 모델 
input1 = Input(shape=(8,))
dense1 = Dense(5)(input1)
dense2 = Dense(7)(dense1)
dense3 = Dense(7)(dense2)
dense4 = Dense(7)(dense3)
dense5 = Dense(7)(dense4)
output1 = Dense(1)(dense4)

model = Model(inputs = input1, outputs = output1)

# 3. 컴파일 훈련 
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss',
              patience =20,
              mode = 'min',
              verbose=1,
              restore_best_weights=True,
              )
model.fit(x_train, y_train,
                 epochs = 100,
                 batch_size = 32,
                 validation_split = 0.2,
                 verbose = 1,
                 # callbacks=[es],
                 )
               


# 4. 평가 예측 

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)
'''
시퀀셜 모델 
loss :  0.5500077605247498
r2 스코어 :  0.5889742933224053
'''
'''
함수 모델
loss :  0.5440099239349365
r2 스코어 :  0.5934564312901275
'''
