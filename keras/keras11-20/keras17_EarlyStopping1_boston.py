# 빨리(Early) 끊는다.(stop)
# 로스가 최소 지점이 최적의 웨이트
# history사용하여 컷시키기
# 소문자는 함수, 대문자는 클라스
# 네이밍 룰  c언어,python(C언어로 만들었다.) , 처음이랑 띄어쓰기 대문자 사용(C언어 계역),_ 쓰는경우(java계열
# 카멜케이스 룰 
# 강제적인건 아님
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets=load_boston()
x=datasets.data
y=datasets['target']
print(x.shape,y.shape) #(506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7, random_state=8715
)

#2. 모델
model=Sequential()
model.add(Dense(2,activation="sigmoid",input_dim=13))
model.add(Dense(4,activation="sigmoid"))
model.add(Dense(8,activation="sigmoid"))
model.add(Dense(16,activation="sigmoid"))
model.add(Dense(32,activation="sigmoid"))
model.add(Dense(64,activation="sigmoid"))
model.add(Dense(32,activation="sigmoid"))
model.add(Dense(16,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(2,activation="relu"))
model.add(Dense(1,activation="linear"))

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100,mode='min',
              verbose=1, 
              restore_best_weights=True
              ) # 브레이크 잡은 시점에서 가중치를 사용해서 예측을 실행함 ( 복원한다고 표현)
#verbose = 끊는 지점을 볼수 있다. 
# val_loss가 나음
# patience = 몇 번까지 참을지
#mode = min - 최소 값을 찾아라 
hist =model.fit(x_train,y_train,epochs=1000,batch_size=800,
          validation_split=0.2,verbose=1,
          callbacks=[es]
          )

# print("---------------------------------------------")
# print("hist:",hist)
# #<tensorflow.python.keras.callbacks.History object at 0x000001B2069A5760>
# print("---------------------------------------------")
# print(hist.history)
# print("---------------------------------------------")
# print(hist.history['loss'])
print("---------------------------------------------")
print(hist.history['val_loss'])
#훈련 했던 값들은  model에 저장되어있다.
#model.fit은 결과치를 반환한다. 


# result =model.predict([17])
# 뭔가 명시하지 않아도 된다는데 
import matplotlib.pyplot as plt
#한글 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['val_loss'],marker='.',c='red',label='val_loss') # 뭔가 명시하지 않아도 된다는데 
plt.plot(hist.history['loss'],marker='.',c='blue',label='loss') # 뭔가 명시하지 않아도 된다는데 
plt.title('보스턴') #이름 지어주기,한글로 쓰면 깨진다. #한글로 제대로 나오게 하는 방법이 있다. 
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend() #
plt.grid() #
plt.show()

# 발로스가 로스보다 더 높은 곳에 있다는거

# 4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss : ',loss)
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print('r2 스코어 :',r2)

