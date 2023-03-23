import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
#1. 데이터
datasets= load_breast_cancer()
print(datasets)
# 사전에서 무언가를 찾는다. 
# 딕셔너리 사전
# 키밸류 형태로 저장한 사전 : 딕셔너리
# 중괄호형태로 되어있다. 
# print(datasets.DESCR) #판다스 : .describe()
# 열 이름 보기 # 판다스 : .columns
#판다스는 많이 사용하기 때문에 이해하기 
#DESCR 찾아보기

x=datasets['data']
y=datasets.target
#위에 둘다 딕셔너리의 키
print(x.shape,y.shape)#(569,30),(569,)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,shuffle=True,random_state=333, test_size=0.2
)

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
# 마지망게 sigmoid 넣는다. ㅣ
# 최종값을 0과 1로 사이로 한정시키다. 
# 활성화함수 

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy','mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=10,mode='min',
               verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=700,batch_size=800,verbose=1,validation_split=0.2,)
# 회귀에서는 매트릭에서 입력해서 볼 수 있다. 
#mse  wl
# 4. 평가, 예측
result =model.evaluate(x_test,y_test) # loss와 메트릭스에 집어넣은 값이 들어간다. 
# loss: 0.3266 - accuracy: 0.8901 - mse: 0.0905 - mae: 0.1515 - val_loss: 0.2226 - val_accuracy: 0.9011 - val_mse: 0.0655 - val_mae: 0.1107
print('result : ',result )
y_predict=np.round(model.predict(x_test))
# 정확도 지표
# 0이나 1이냐 
# accuracy_score : 서로 같냐를 따지는 지표
# print('--=-===================================================')
# print(y_test[:5]) #010101
# print(y_predict[:5]) #실수 형태
# print(np.round(y_predict)) #반올림
# 실수형태에서 0또는 1로 한정시키고 싶다. 
# accuracy_score에서 꽝이다. 
# Classification metrics can't handle a mix of binary and continuous targets
# 0101(이진분류)과 연속 숫자와 같이 처리할 수 없어...ㅠㅜ 
#반올림해도 된다.
# continuous targets: 010101  이런 데이터
acc = accuracy_score(y_test,y_predict)
# acc = accuracy_score(y_test,y_predict)
print('acc : ',acc)
# acc :  0.8947368421052632 #89프로 맞춤
# 분류의 종류 이중 분류 / 다중분류

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='blue',label='loss') # 뭔가 명시하지 않아도 된다는데 
plt.title('보스턴') #이름 지어주기
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()
