# 2개이상은 다중분류 
# 다중분류 안에 이진분류가 있다. 
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score #평가용 #r2는 회귀, accuracy는 분류모델에서 사용하고 가장 디폴트적인 것


#1. 데이터
datasets = load_iris()
print(datasets.DESCR) #판다스 describe()
print(datasets.feature_names) # 판다스 clolums
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x=datasets['data']
y=datasets.target
print(x.shape,y.shape) #(150, 4) (150,)
print(x)
print(y) 4⁴ㅡ
# 왜 섞어줘야하지?????????????
# 와이 값큰 것은 라벨값을 알기 위해서 어떻게 해야하나
# 넘파이에 갯수를 알 수 있는 것이 있다. 
print("y의 라벨 값 :",np.unique(y)) #[0 1 2]
############요지점에서 원핫을 해야겠죠?##############
#나누기전 원핫! 판다스/사이킷런/텐서플로우에 각각존재
# 텐서플로우  tf.one_hot() tf.one_hot(라벨 인덱스 목록, 라벨 개수
y = to_categorical(y)
# 판다스 pd.get_dummies() 투 카테코리컬
# 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# train_cat = ohe.fit_transform(train[['cat1']])
한 ㅛ################################################

x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    train_size=0.8,
    shuffle=True,random_state=456,
    stratify= y 
)# 이론상1으로는 이게 맞긴하지만 가끔은 안넣었을 때 잘 나올수도 있다. 
    # 통계적으로 해라
print("y_test : ",y_test)
print(np.unique(y_train,return_counts=True))
# y 라벨 값이 특정 값에 몰려있으면 그 특정값이 될 확률 이 올리간다. 
# 분류 모델에서는 와이라벨값이 예측값에 영향을 미칠 수 있다. 
# 방지를 위해서 비율이 1:1 비슷한 수준으로는 되야한다. (5.5:4.5 처럼)

# 2. 모델 구성
model =Sequential()
model.add(Dense(50,activation="relu",input_dim=4))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(3,activation="softmax")) #반드시 이것만
#라벨의 클라스의 갯수, 라벨 갯수 
#(32, 1) and (32, 3)
# 둘이 벡터가 달라서 문제가 생김 
#원핫 인코딩 - 내 추측 : 하나의 벡터를 만들어서 나오게 만들기
##원핫 인코딩 선생님의 설명:  값의 가치를 먹이지 말고 위치 값을 먹이자
#각각의 라벨에 대한 확률값이 필요해서 3개가 필요하다. 
#리니어일때는 1개의 값, 시그모이드는 0이나 1의 값, 소프트 맥스는 라벨 값 만큼 노드를 뽑느다. 
#값이 0.5 0.4 0.1일때 값이 가장 높은 0.5가 ㅇㅋ이다. 
# 다중분류에서는 소프트맥스 
# 라벨이 3개여도 합이 1이된다.
# 0,1,2라고 표현을 했지만, 가치가 다르다는건 아닌데
#원핫 인코딩 선생님의 설명:  값의 가치를 먹이지 말고 위치 값을 먹이자
#라벨명 위치값       가치값
# 가위 [1,0,0] 1+0+0 = 1
# 바위 [0,1,0] 0+1+0 = 1
#  보  [0,0,1] 0+0+1 = 1
# 예측값은 [0.1,0.7,0.2] 가장 높은게 [0,1,0]
# 예측값은 [0.5,0.2,0.3] 가장 높은게 [1,0,0]
# 0 ~ 1사이 

# 3. 컴파일 훈련 
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=10,mode='max',
               verbose=1,restore_best_weights=True)
hist=model.fit(x_train,y_train,epochs=700,batch_size=800,verbose=1,validation_split=0.2,)




# 4. 평가, 예측
result =model.evaluate(x_test,y_test) 
print('result : ',result )
print('loss : ',result[0] )
print('acc : ',result[1] )
y_predict=model.predict(x_test) #위치에 대한 반환
y_test_acc = np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=-1)
print('y_test_acc : ',y_test_acc)
print('y_predict : ',y_predict )
acc = accuracy_score(y_test_acc,y_predict)
#y_test_acc도 아래 같은 식으로 만든다.
#y_predict : [0 1 2 2 0 1 2 1 2 2 2 0 2 2 0 1 1 0 1 0 0 0 0 1 2 2 2 1 1 0] 
print('accuracy_score : ',acc)
"""
result :  [0.21222148835659027, 0.9666666388511658]
acc :  0.966666666666666    
"""
# [과제]튜플
# [과제] accuravy_score를 사용해서 스코어를 빼세요
# 가장 높은걸 1로해달라는 걸 넘파이에 있다. 사용해라
# np.argmax | axis =  엑시즈 = 1 = 행에 맞춰서
