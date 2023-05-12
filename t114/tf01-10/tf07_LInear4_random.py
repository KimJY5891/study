import tensorflow as tf

tf.set_random_seed(337)

# 1. 데이터

x=[1,2,3,4,5]
y = [2,4,6,8,10]

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 
# w = tf.random_normal([1]) # 가능 
# b = tf.random_normal([1]) # 가능 
# tf.random_normal() : 정규분포
# tf.random_uniform() : 엔빵
# b = tf.Variable(100,dtype=tf.float32) # 바이어스
# 웨이트값이 고정값보다 랜덤값이 더 좋다. 
# 어차피 모르는거 내가 하는거보다 랜덤이 나을지도
'''
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w)) # [-0.4121612]
# 위와 동일 
with tf.compat.v1.Session() as sess :
    sess.run(tf.global_variables_initializer())
    print(sess.run(w)) # [-0.4121612]
'''
# 경사하강법 방법이라 웨이트를 크게 줘도 알아서 줄어든다. 
 
# 2. 모델 구성

hypothesis = x * w + b

# 3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
epochs = 2000
for step in range(epochs) :
    sess.run(train)
    if step %20 == 0 :
        print(step, sess.run(loss),sess.run(w),sess.run(b))
sess.close() # 하지 않으면 메모리 차지할 수도 있다. 

# 4. 평가 예측


