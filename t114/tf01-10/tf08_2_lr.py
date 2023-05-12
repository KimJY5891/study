# 오류 수정요망

import tensorflow as tf

tf.set_random_seed(337)

# 1. 데이터

x = tf.placeholder(tf.float32,shape=[None])
y = tf.placeholder(tf.float32,shape=[None])

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 

# 2. 모델 구성

hypothesis = x * w + b

# 3-1. 컴파일 

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 줄이는게 좋을 수도 있고 아닐 수도 있음음 
# 로컬 미니멈 가짜 최적의 가중치
# 러닝레이트가 크면 성큼성큼 떨어진다. 
train = optimizer.minimize(loss)

#3-2. 훈련

with tf.compat.v1.Session() as sess  :  # 
    sess.run(tf.global_variables_initializer())
    epochs = 2001
    for step in range(epochs) :
        # sess.run(train)
        _,loss_val,w_val, b_val = sess.run([train,loss,w,b], feed_dict = {x:[1,2,3,4,5], y : [2,4,6,8,10]})
        if step %20 == 0 :
            print(step, loss_val,w_val,b_val) 
            
#################### [실습] #########################

#예측값을 뽑아라
 
#####################################################
y_pred = tf.placeholder(tf.float32,shape=[None])
x_data = [6,7,8]

with tf.compat.v1.Session() as sess  :
        # sess.run(tf.global_variables_initializer())# 초기화
        # y_pred = x_data * w_val + b_val
        # y_pred = sess.run(y_pred,feed_dict={x_data:x_data
        #                                     #b_val:b_val,w_val:w_val
        #                                     })
        # print(step,'번째 :', y_pred)
    # [정답]  
    x_test = tf.compat.v1.placeholder(tf.float32,shape=[None])       
    y_pred = x_test * w_val + b_val
    print(sess.run(y_pred),feed_dict={x_test:x_data})
