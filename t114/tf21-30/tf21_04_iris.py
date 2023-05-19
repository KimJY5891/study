import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score #평가용 #r2는 회귀, accuracy는 분류모델에서 사용하고 가장 디폴트적인 것
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리


#1. 데이터
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
x=datasets['data']
y=datasets.target
y = y.reshape(-1,1)
print(x.shape,y.shape) #(150, 4) (150,)
print("y의 라벨 값 :",np.unique(y)) #[0 1 2]

x_train, x_test, y_train, y_test = train_test_split(
    x,y, 
    train_size=0.8,
    shuffle=True,random_state=456,
    stratify= y 
)
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
xp = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,50]),name='weight')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([50]),name='bias')
layer1= tf.compat.v1.matmul(xp,w1) +b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([50,40]),name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([40]),name='bias')
layer2= tf.compat.v1.matmul(layer1,w2) +b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([40,30]),name='weight')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([30]),name='bias')
layer3= tf.compat.v1.matmul(layer2,w3) +b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,20]),name='weight')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([20]),name='bias')
layer4= tf.compat.v1.matmul(layer3,w4) +b4

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,10]),name='weight')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias')
layer5=  tf.compat.v1.matmul(layer4,w5) +b5

w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,3]),name='weight')
b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([3]),name='bias')
layer6=  tf.compat.v1.matmul(layer5,w6) +b6

w7 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),name='weight')
b7 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer6,w7) +b7) 
'''
model =Sequential()
model.add(Dense(50,activation="relu",input_dim=4))
model.add(Dense(40,activation="relu"))
model.add(Dense(30,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(3,activation="softmax"))
'''

# 3-1. 컴파일 
loss = tf.reduce_mean(tf.reduce_sum(yp*tf.log(hypothesis),axis=1))
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-5).minimize(loss)

# 3-2. 훈련 

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 101
for step in range(epochs) :
    _, cost_val = sess.run([train,loss],feed_dict={xp:x_train,yp:y_train})
    if step % 20 == 0 :
        print("Epoch:", step, "Loss:", cost_val)
'''
Epoch: 0 Loss: 0.0
Epoch: 20 Loss: 0.0
Epoch: 40 Loss: 0.0
Epoch: 60 Loss: 0.0
Epoch: 80 Loss: 0.0
Epoch: 100 Loss: 0.0
'''
