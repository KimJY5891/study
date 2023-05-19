import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(33777)
x_data = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
],dtype=np.float32)
y_data = np.array([
    [0],[1],[1],[0]
],dtype=np.float32)

# 2. 모델

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])
# model.add(Dense(10,input_shape=(2,)))

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,10]),name='weight')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias')
layer1= tf.compat.v1.matmul(x,w1) +b1
# model.add(Dense(7))
# 중간에 노드가 10개 라면 열이 10개가 된다.

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,16]),name='weight')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]),name='bias')
layer2= tf.compat.v1.matmul(layer1,w2) +b2
# model.add(Dense(7))
# 위에 layer1 결과 값을 받아서 계산하는 것 

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,32]),name='weight')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]),name='bias')
layer3= tf.compat.v1.matmul(layer2,w3) +b3

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,16]),name='weight')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]),name='bias')
layer4= tf.compat.v1.matmul(layer3,w4) +b4

w6 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,8]),name='weight')
b6 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]),name='bias')
layer6=  tf.compat.v1.matmul(layer4,w6) +b6

w7 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]),name='weight')
b7 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer6,w7) +b7)   # 최종이니까 
# model.add(Dense(1,activation='sigmoid'))

# 마지막에 아웃풋은 시그모이드
# w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),name='weight')
# b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')
# layer1= tf.compat.v1.matmu1(x,w1) +b1
# # 앞에 값을 10개 받아서 행이 10개 (2,10)x(10,1)
# # 이것이 최종일 경우 하나의 노드만 가지기 때문에 열이 1개 

# 3-1.컴파일 
cost = tf.reduce_mean(y*tf.log1p(hypothesis) + (1-y)*tf.log1p(1-hypothesis)) # 바이너리컬 센트로피 
# cost = tf.reduce_mean(y*tf.log1p(hypothesis) + (1-y)*tf.log1p(1-hypothesis)) # 바이너리컬 센트로피 
# cost = tf.reduce_mean(y*tf.log_sigmoid(hypothesis) + (1-y)*tf.log_sigmoid(1-hypothesis)) # 바이너리컬 센트로피 
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 캐스팅 자료 형을 바꿔줘라 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32)) # 액큐러시의 수식 

# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 500
for epochs in range(epochs) : 
    cost_val, _, = sess.run([cost,train],feed_dict={x:x_data,y:y_data})
    # cost_val, _, w_val,b_val  = sess.run([loss,train,w,b],feed_dict={x:x_data,y:y_data})
    if epochs % 200 == 0 : # 200번 마다 한 번씩 보는 것 
        print("Epoch:", epochs, "Loss:", cost_val)
h,p,a=sess.run([hypothesis,predicted,accuracy],feed_dict={x:x_data,y:y_data})
print("원래값 : ",y_data,"\n 예측값 : ",p,"\n accuracy : ",a,)

'''
Epoch: 0 Loss: -0.44843686
Epoch: 200 Loss: -0.5032044
Epoch: 400 Loss: -0.5032044
'''
