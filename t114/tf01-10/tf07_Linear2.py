import tensorflow as tf

tf.set_random_seed(337)

# 1. 데이터

x=[1,2,3,4,5]
y = [2,4,6,8,10]

w = tf.Variable(333,dtype=tf.float32) # 가중치
b = tf.Variable(111,dtype=tf.float32) # 바이어스

####### 실습 ####### 만들자.

# 2. 모델 구성

hypothesis = x * w +b

# 3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
epochs = 200
for step in range(epochs) :
    sess.run(train) # 
    
    if step %20 == 0 :
        print(step, sess.run(loss),sess.run(w),sess.run(b)) # 사실상 중복실행
        # 
sess.close()

