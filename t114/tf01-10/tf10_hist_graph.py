# 오류 수정요망
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
train = optimizer.minimize(loss)

#3-2. 훈련
loss_val_list = []
w_val_list = []
b_val_list = []
with tf.compat.v1.Session() as sess  :  # 
    sess.run(tf.global_variables_initializer())
    epochs = 101
    for step in range(epochs) : # 한 번이 1 훈련
        # sess.run(train)
        _,loss_val,w_val, b_val = sess.run([train,loss,w,b], feed_dict = {x:[1,2,3,4,5], y : [2,4,6,8,10]})
        # 히스토리를 만들고 싶으면 리스트형태로 만들면 된다. 
        # 에포 단위로 각 값을 리스트에 모은다. 
        
        if step %20 == 0 :
            print(step, loss_val,w_val,b_val) 
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        b_val_list.append(b_val)
        
print(loss_val_list)

import matplotlib.pyplot as plt
plt.subplot(2,2,1)
plt.scatter(w_val_list,loss_val_list)
plt.xlabel('w')
plt.ylabel('loss')


plt.subplot(2,2,2)
plt.plot(w_val_list,loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')

plt.subplot(2,2,3)
plt.plot(w_val_list)
plt.xlabel('epochs')
plt.ylabel('w')

plt.subplot(2,2,4)
plt.plot(b_val_list)
plt.xlabel('epochs')
plt.ylabel('b')
plt.show()
