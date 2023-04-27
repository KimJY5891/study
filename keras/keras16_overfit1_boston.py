from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#1. 데이터
datasets=load_boston()
x=datasets.data
y=datasets['target']
print(x.shape,y.shape) #(506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.625, random_state=8715
)

#2. 모델
model=Sequential()
model.add(Dense(2,input_dim=13))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam')
hist =model.fit(x_train,y_train,epochs=100,batch_size=120,
          validation_split=0.2,verbose=1)
print(hist.history)
#훈련 했던 값들은  model에 저장되어있다.
#model.fit은 결과치를 반환한다. 

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',loss)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic' #한글  
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

