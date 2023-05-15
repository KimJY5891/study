# 오류 수정요망
import tensorflow as tf

tf.set_random_seed(337)

# 1. 데이터

x = tf.placeholder(tf.float32,shape=[None])
y = tf.placeholder(tf.float32,shape=[None])

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 

#####[실습]#####
# 2. 모델 구성

hypothesis = x * w + b

# 3-1. 컴파일 

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련

with tf.compat.v1.Session() as sess  :  
# sess = tf.compat.v1.Session() with가 이거 대신
    sess.run(tf.global_variables_initializer())
    epochs = 2001
    for step in range(epochs) :
        # sess.run(train)
        _,loss_val,w_val, b_val = sess.run([train,loss,w,b], feed_dict = {x:[1,2,3,4,5], y : [2,4,6,8,10]})
        if step %20 == 0 :
            print(step, 'loss :', loss_val, 'w :', w_val, 'b :', b_val)
            
#################### [실습] #########################

#예측값을 뽑아라
 
#####################################################

'''
x_data = tf.placeholder(tf.float32,shape=[None])
y_pred = tf.placeholder(tf.float32,shape=[None])

with tf.compat.v1.Session() as sess  :
    for step, x_data in enumerate(x_data) :
        sess.run(tf.global_variables_initializer())# 초기화
        y_pred = x_data * w_val + b_val
        y_pred = sess.run(y_pred,feed_dict={x_data:[6,7,8]
                                            #b_val:b_val,w_val:w_val
                                            })
        # tensor도 자료형태의 하나이다. 
        print(step,'번째 :', y_pred)
'''

# [정답]  
with tf.compat.v1.Session() as sess  :
    x_test = tf.compat.v1.placeholder(tf.float32,shape=[None])       
    y_pred = x_test * w_val + b_val
    y_pred = sess.run([y_pred], feed_dict={x_test:[6,7,8]})
    print('y_predict :', y_pred[0][0], y_pred[0][1], y_pred[0][2])
'''
RuntimeError: Attempted to use a closed Session.
tensorFlow의 세션(Session)이 이미 닫혀있는 상태에서
세션을 사용하려고 할 때 발생합니다. TensorFlow의 세션은 사용이 끝나면
명시적으로 닫아주어야 합니다. 세션을 닫지 않은 상태에서 
다시 세션을 사용하려고 하면 이러한 예외가 발생할 수 있습니다.
'''


