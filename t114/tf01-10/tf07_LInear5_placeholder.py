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
train = optimizer.minimize(loss)

#3-2. 훈련

with tf.compat.v1.Session() as sess  :  # 
    sess.run(tf.global_variables_initializer())
    epochs = 2001
    for step in range(epochs) :
        # sess.run(train)
        _,loss_val,w_val, b_val = sess.run([train,loss,w,b], feed_dict = {x:[1,2,3,4,5], y : [2,4,6,8,10]})
        if step %20 == 0 :
            print(step, loss_val,w_val,b_val) # 또 사용하려면 위에 사용해서 초기화를 해줘야해서 이미 부여한 변수 명을 넣어주는 것이다.
# with 문법일 경우 close 하지 않아도 알아서 close 된다. 
# 텐서 2도 텐서 1처럼 그래프 형식인데, 즉시 실행 방식으로 해서 보여주는 것 뿐이다. 
