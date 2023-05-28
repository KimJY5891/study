# pip install keras == 1.2.2
# from tensorflow.keras.datasets import mnist # 1.14버전에 가능하긴함 준비중이었음 
from keras.datasets import mnist
import keras
print(keras.__version__)
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np
import time

# Using TensorFlow backend.
# 텐서플로우 2.-보다 느림 
# 왜냐면 텐서플로우 위에 케라스를 사용하는 것이라 느림 
# 텐서플로우 2.-부터 케라스가 텐서플로우 안에 들어가서 더 빨라짐

tf.set_random_seed(337) # 1.-버전 가능 
# tf.random.set_seed(337) # 2.-버전 가능 

# 1. 데이터 구성

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)-> (60000, 28 * 28) (10000, 28 *28)
print(y_train.shape,y_test.shape) # (60000,) (10000,) -> (60000, 10) (10000, 10)

# 2. 모델 구성 

x = tf.compat.v1.placeholder('float',[None,28,28,1]) # 연산되는 것도 4차원 
y = tf.compat.v1.placeholder('float',[None,10])

# w1 = tf.get_variable('w1',shape =[
    # 3,3, = kerel_size
    # 1, = channel
    # 32 = filters]) 
w1 = tf.get_variable('w1',shape =[3,3,   1,                64]) 
#model.add(Conv2D(64,kerel_size = (3,3),channels(아웃풋) filters
b1 = tf.Variable(tf.zeros([64]))
# layer1 = tf.compat.v1.matmul(x,w1) + b1 # model.add(Dense())
layer1 = tf.compat.v1.nn.conv2d(x,w1,strides=[1,1,1,1,],padding='SAME') # model.add(Conv2D())
# strides=[1,1,1,1,] => 한칸씩 이동한다. 즉 1이다. 앞뒤는 4차원이라서 맞춰주는것 
# [1,2,2,1] = > 한 칸 더 전진하고 싶은면 가운데만 맞추면 된다. 
layer1 += b1
L1_maxpool = tf.nn.max_pool2d(layer1,ksize=[1,2,2,1],strides=[1,2,2,1,],padding='SAME')  # 반띵을 위함 
# ksize => 

w2 = tf.Variable(tf.random_normal([3,3,64,32]))
b2 = tf.Variable(tf.zeros([32]))
layer2 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(L1_maxpool,w2) )
layer2 += b2
L2_maxpool = tf.nn.max_pool2d(layer2,ksize=[1,2,2,1],strides=[1,2,2,1,],padding='VALID')  # 반띵을 위함 
# (n,6,6,32)


w3= tf.Variable(tf.random_normal([3,3,32,16]))
b3 = tf.Variable(tf.zeros([16]))
layer3 = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(L2_maxpool,w2),strides=[1,2,2,1,] ,padding='SAME' )
layer3 += b3
# (n,6,6,16)

# flatten
L_flat = tf.reshape(layer3,[-1,6*6*16])

# 레이어4 Dnn
w4= tf.Variable(tf.random_normal([6*6*16,100]))
b4 = tf.Variable(tf.zeros([100]))
layer4 = tf.nn.relu(tf.compat.v1.matmul(layer3,w4)+b4)
layer4 = tf.nn.dropout(L_flat,rate=0.3)


w5= tf.Variable(tf.random_normal([6*6*16,100]))
b5 = tf.Variable(tf.zeros([100]))
hypothesis = tf.compat.v1.matmul(layer4,w5)+b5
hypothesis = tf.nn.softmax(hypothesis)

# 3-1. 컴파일
# cost = tf.reduce_mean(tf.reduce_sum(y*tf.log(hypothesis),axis=1))
cost = tf.reduce_mean(tf.reduce_sum(y*tf.nn.log_softmax(hypothesis),axis=1))
# cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,lavels=y))
# cost = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

start_time = time.time()
# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 100
batch_size = 6000
total_batch = int(len(x_train)/batch_size) # 60000/100 = 600
for epochs in range(epochs) : 
    for i in range(total_batch) :  # 100개씩 600번 돌자 
        start = i *batch_size       
        end  = start + batch_size  
        cost_val, _,w_val,b_val = sess.run([cost,train,w4,b4],feed_dict={x:x_train[start:end],y:y_train[start:end]})
        avg_cost = cost_val/total_batch
    print("Epoch : ", epochs+1, "Loss : {:.9f}".format(avg_cost))
end_time = time.time()
    
print('훈련끝')        


# 4. 평가 훈련 
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
y_pred = sess.run(hypothesis,feed_dict= {x:x_test})
y_pred_arg = sess.run(tf.argmax(y_pred,axis=1))
print(type(y_test))
print(type(y_pred_arg))

acc = accuracy_score(y_test,y_pred_arg) 
print('acc : ',acc)
mae = mean_absolute_error(y_test,y_pred_arg)
print('mae : ',mae)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 캐스팅 자료 형을 바꿔줘라 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32)) # 액큐러시의 수식 

a=sess.run([accuracy],feed_dict={x:x_test,y:y_test})
print("accuracy : ",acc)
print("tf",tf.__version__,'걸린 시간 : ',round(end_time-start_time,2))

sess.close()
