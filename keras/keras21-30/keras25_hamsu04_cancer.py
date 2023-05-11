from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing,load_diabetes, load_breast_cancer
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test,y_train,y_test = train_test_split(
    x,y, test_size=0.2,shuffle=True,random_state=337
)
print(x_train.shape)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

# 시퀀셜 모델 
model = Sequential()
model.add(Dense(5, input_dim = 30))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

# 함수 모델 
# input1 = Input(shape=(30,))
# dense1 = Dense(5)(input1)
# dense2 = Dense(7)(dense1)
# dense3 = Dense(7)(dense2)
# dense4 = Dense(7)(dense3)
# dense5 = Dense(7)(dense4)
# output1 = Dense(1)(dense4)

# model = Model(inputs = input1, outputs = output1)

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
loss :  0.06003603711724281
r2 스코어 :  0.7081076257139403
'''
'''
함수 모델
loss :  0.06181302294135094
r2 스코어 :  0.6994679905584945
'''
# 결과 시퀀셜 모델 승
