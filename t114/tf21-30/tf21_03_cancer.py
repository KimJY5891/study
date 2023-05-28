# sigmoid
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
y = y.reshape(-1,1)

print(x.shape,y.shape)#(569,30),(569,1)
# print(y)

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
xp = tf.compat.v1.placeholder(tf.float32,shape=[None,30])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,10]),name='weight')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias')
layer1= tf.compat.v1.matmul(xp,w1) +b1
# model.add(Dense(100))
# 중간에 노드가 10개 라면 열이 10개가 된다.

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,24]),name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([24]),name='bias')
layer2= tf.compat.v1.matmul(layer1,w2) +b2
# model.add(Dense(7))
# 위에 layer1 결과 값을 받아서 계산하는 것 

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([24,80]),name='weight')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([80]),name='bias')
layer3= tf.compat.v1.matmul(layer2,w3) +b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([80,40]),name='weight')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([40]),name='bias')
layer4= tf.compat.v1.matmul(layer3,w4) +b4

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([40,20]),name='weight')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([20]),name='bias')
layer5=  tf.compat.v1.matmul(layer4,w5) +b5

w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,10]),name='weight')
b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias')
layer6=  tf.compat.v1.matmul(layer5,w6) +b6

w7 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),name='weight')
b7 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer6,w7) +b7) 

'''
model =Sequential()
model.add(Dense(10,activation="relu",input_dim=30))
model.add(Dense(24,activation="linear"))
model.add(Dense(80,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(1,activation='sigmoid'))
'''

# 3-1. 컴파일 
cost = tf.reduce_mean(yp*tf.log_sigmoid(hypothesis)+(1-yp)*tf.log_sigmoid(1-hypothesis)) # binary_crossentropy
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 101
for step in range(epochs) : 
    cost_val, _ = sess.run([cost,train],feed_dict = {xp:x_train,yp:y_train})
    if step % 20 == 0 : 
        print("Step:", step, "Loss:", cost_val)
sess.close()
'''
ed to use: AVX2
Step: 0 Loss: -0.4334891
Step: 20 Loss: -0.4334891
Step: 40 Loss: -0.4334891
Step: 60 Loss: -0.4334891
Step: 80 Loss: -0.4334891
Step: 100 Loss: -0.4334891
''' 
