import tensorflow as tf
import numpy as np
tf.set_random_seed(337)
x_data = [
    [1,2,1,1],
    [2,1,3,2],
    [3,1,3,4],
    [4,1,5,5],
    [1,7,5,5],
    [1,2,5,6],
    [1,6,6,6],
    [1,7,6,7],
]
y_data = [
    [0,0,1], #2
    [0,0,1],
    [0,0,1],
    [0,1,0], #1
    [0,1,0],
    [0,1,0],
    [1,0,0], # 0
    [1,0,0],
]
# 2. 모델 구성 
x= tf.compat.v1.placeholder(tf.float32,shape=[None,4])
w = tf.compat.v1.Variable(tf.random_normal([4,3]))
b= tf.compat.v1.Variable(tf.zeros([3]))
y = tf.compat.v1.placeholder(tf.float32,shape=[None,3])

hypothesis = tf.compat.v1.matmul(x,w)+b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-5).minimize(loss)
# optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-5)
# train =optimizer.minimize(loss)
# 합친것

# 3-2. 훈련 

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 101
for step in range(epochs) :
    _, cost_val, w_val,b_val = sess.run([train,loss,w,b,],feed_dict={x:x_data,y:y_data})
    if step % 20 == 0 :
        print("Epoch:", epochs, "Loss:", cost_val)
        
# 4. 평가 훈련 
from sklearn.metrics import r2_score, mean_absolute_error
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
y_pred = tf.matmul(x_test,w_val)  +b_val
y_aaa = sess.run(y_pred,feed_dict={x_test:x_data})

r2 = r2_score(y_data,y_aaa)
print('r2 : ',r2)
mae = mean_absolute_error(y_data,y_aaa)
print('mae : ',mae)
sess.close()
