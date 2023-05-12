import tensorflow as tf
tf.compat.v1.set_random_seed(337)

변수  = tf.compat.v1.Variable(tf.random_normal([2]),name='weight')

# 초기화 첫 번째

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수) 
print('aaa : ',aaa)
sess.close() 

# 초기화 두 번째

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb =  변수.eval(session=sess) 
sess.close() 

# 초기화 세 번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
sess.close()


# 1. 데이터

x = tf.placeholder(tf.float32,shape=[None])
y = tf.placeholder(tf.float32,shape=[None])


# [실습]
# 08_2를 카피 해서 아래 만들어보기


################################## 1. Session() //sess.run(변수)

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
w= sess.run(w)
b = sess.run(b)
print('w : ' , w )
print('b : ' , b )
sess.close() 

################################## 2. Session() //변수.eval(session=sess)
w = tf.Variable(tf.random_normal([1]),dtype=tf.float32) 
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
w = w.eval(session=sess)
b = b.eval(session=sess)
print('w : ' , w )
print('b : ' , b )
sess.close() 


################################## 3. InteractiveSession // 변수.eval()

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
w = w.eval()
b = b.eval()
print('w : ' , w )
print('b : ' , b )
sess.close()


# 2. 모델 구성

hypothesis = x * w + b

# 3-1. 컴파일 

loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


# 3-2. 훈련 
sess = tf.compat.v1.Session() 
