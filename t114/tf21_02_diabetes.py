
# keras23스케일러를 보스턴 빼고 
# 캘리포니아
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

#1. 데이터

datasets= load_diabetes()
x= datasets.data
y= datasets.target
y = y.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)
print(y)

#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("x_train:",x_train.shape) #(397, 10)
print("y_train:",y_train.shape) #(397,1)

#2.모델 구성

xp = tf.compat.v1.placeholder(tf.float32,shape=[None,10])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,100]),name='weight')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([100]),name='bias')
layer1= tf.compat.v1.matmul(xp,w1) +b1
# model.add(Dense(100))
# 중간에 노드가 10개 라면 열이 10개가 된다.

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([100,80]),name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([80]),name='bias')
layer2= tf.compat.v1.matmul(layer1,w2) +b2
# model.add(Dense(7))
# 위에 layer1 결과 값을 받아서 계산하는 것 

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80,64]),name='weight')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]),name='bias')
layer3= tf.compat.v1.matmul(layer2,w3) +b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,40]),name='weight')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([40]),name='bias')
layer4= tf.compat.v1.matmul(layer3,w4) +b4

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([40,24]),name='weight')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([24]),name='bias')
layer5=  tf.compat.v1.matmul(layer4,w5) +b5

w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([24,16]),name='weight')
b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]),name='bias')
layer6=  tf.compat.v1.matmul(layer5,w6) +b6

w7 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,8]),name='weight')
b7 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]),name='bias')
layer7=  tf.compat.v1.matmul(layer6,w7) +b7

w8 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]),name='weight')
b8 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')
hypothesis = tf.compat.v1.matmul(layer7,w8) +b8   # 최종이니까 
'''
아래 회귀 모델의 위와 같이 수정 
model.add(Dense(100,input_dim=10))
model.add(Dense(80))
model.add(Dense(64))
model.add(Dense(40))
model.add(Dense(24))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))
'''

# 3-1. 컴파일 
# cost = tf.reduce_mean(tf.square(hypothesis-y)) # 이럴 경우, y의 값들이 들어오고 y 플레이스 홀의 변수자리인 yp를 작성해줘야한다. 예전 다른 코드에서도 y는 사실상 플레이홀 형태임
cost = tf.reduce_mean(tf.square(hypothesis-yp)) 
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# 3-2. 훈련 
epochs = 2001
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for step in range(epochs) : 
    cost_val, _ = sess.run([cost,train],feed_dict = {xp:x_train,yp:y_train})
    if step % 20 == 0 : 
        print("Step:", step, "Loss:", cost_val)
sess.close()
        
