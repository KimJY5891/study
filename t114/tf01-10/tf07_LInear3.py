import tensorflow as tf

tf.set_random_seed(337)

# 1. 데이터
x=[1,2,3]
y = [1,2,3]
w = tf.Variable(111,dtype=tf.float32) # 가중치
b = tf.Variable(100,dtype=tf.float32) # 바이어스

# 2. 모델 구성 

hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


# 3-2. 훈련
with tf.compat.v1.Session() as sess  :  # 
# sess = tf.compat.v1.Session() 이것이 대체 된 것이다. 
    sess.run(tf.global_variables_initializer())# 초기화
    epochs = 2001
    for step in range(epochs) : 
        sess.run(train)
        if step %20 == 0 : 
            print(step,sess.run(loss),sess.run(w),sess.run(b)) # verbose
    # sess.close()  이거 없이도 위드문에 끝나면 클로즈 같은 효과가 난다. 


# 4. 평가 예측

