# pip install keras == 1.2.2
# from tensorflow.keras.datasets import mnist # 1.14버전에 가능하긴함 준비중이었음 
from keras.datasets import mnist
import keras
print(keras.__version__)
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np
# Using TensorFlow backend.
# 텐서플로우 2.-보다 느림 
# 왜냐면 텐서플로우 위에 케라스를 사용하는 것이라 느림 
# 텐서플로우 2.-부터 케라스가 텐서플로우 안에 들어가서 더 빨라짐
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])/255.
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])/255.
print(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)-> (60000, 28 * 28) (10000, 28 *28)
print(y_train.shape,y_test.shape) # (60000,) (10000,) -> (60000, 10) (10000, 10)

# [실습] 만들기 

# 2. 모델

xp = tf.compat.v1.placeholder(tf.float32,shape=[None,x_train.shape[1]])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,y_train.shape[1]])


w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_train.shape[1],10]),name='weight')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias')
layer1= tf.compat.v1.matmul(xp,w1) +b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,16]),name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]),name='bias')
layer2= tf.compat.v1.matmul(layer1,w2) +b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,32]),name='weight')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]),name='bias')
layer3= tf.compat.v1.matmul(layer2,w3) +b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,16]),name='weight')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]),name='bias')
layer4= tf.compat.v1.matmul(layer3,w4) +b4

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,8]),name='weight')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]),name='bias')
layer5=  tf.compat.v1.matmul(layer4,w5) +b5

w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,y_train.shape[1]]),name='weight')
b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([y_train.shape[1]]),name='bias')
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer5,w6) +b6)


# 3-1.컴파일 

cost = tf.reduce_mean(tf.reduce_sum(yp*tf.log(hypothesis),axis=1))
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-5).minimize(cost)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)


# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 100
for epochs in range(epochs) : 
    cost_val, _, = sess.run([cost,train],feed_dict={xp:x_train,yp:y_train})
    # cost_val, _, w_val,b_val  = sess.run([loss,train,w,b],feed_dict={x:x_data,y:y_data})
    if epochs % 2 == 0 : # 200번 마다 한 번씩 보는 것 
        print("Epoch:", epochs, "Loss:", cost_val)

# 4. 평가 훈련 
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
y_pred = tf.matmul(x_test,w6)+b6
print('y_pred',y_pred)
y_aaa = sess.run(y_pred,feed_dict={x_test:x_test})
print('y_aaa',y_aaa)
y_aaa = np.argmax(y_aaa,axis=1)
print('y_aaa',y_aaa)

acc = accuracy_score(y_test,y_aaa) 
print('acc : ',acc)
mae = mean_absolute_error(y_test,y_aaa)
print('mae : ',mae)
sess.close()
