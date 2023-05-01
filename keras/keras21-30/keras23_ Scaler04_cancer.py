from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets= load_breast_cancer()
print(datasets)
x=datasets['data']
y=datasets.target

print(x.shape,y.shape)#(569,30),(569,)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,shuffle=True,random_state=333, test_size=0.2
)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model =Sequential()
model.add(Dense(10,activation="relu",input_dim=30))
model.add(Dense(8,activation="linear"))
model.add(Dense(4,activation="linear"))
model.add(Dense(2,activation="linear"))
model.add(Dense(4,activation="relu"))
model.add(Dense(8,activation="linear"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(80,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(8,activation="linear"))
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
es = EarlyStopping(monitor='val_accuracy',patience=50,mode='max',
               verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=1400,batch_size=800,verbose=1,validation_split=0.2,)

# 4. 평가, 예측
result =model.evaluate(x_test,y_test)
print('result : ',result )

y_predict=np.round(model.predict(x_test))
acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)


"""
NON
result :  [0.6360973715782166, 0.6754385828971863, 0.22190581262111664]
acc :  0.6754385964912281
"""
"""
scaler = StandardScaler()
result :  [1.570584774017334, 0.9473684430122375, 0.05121266096830368]
acc :  0.9473684210526315
"""
"""
#scaler = MinMaxScaler()
result :  [0.885256826877594, 0.9561403393745422, 0.043656762689352036]
acc :  0.956140350877193
"""
"""
#scaler = MaxAbsScaler()
result :  [1.4353965520858765, 0.9649122953414917, 0.025789322331547737]
acc :  0.9649122807017544
"""
"""
scaler = RobustScaler()
result :  [1.305980920791626, 0.9385964870452881, 0.06139994412660599]
acc :  0.9385964912280702
"""
# StandardScaler 승
