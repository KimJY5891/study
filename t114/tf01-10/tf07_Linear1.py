import tensorflow as tf

tf.set_random_seed(337)

# 1. 데이터
x = [1,2,3]
y = [1,2,3]
w = tf.Variable(111,dtype=tf.float32) # 가중치
b = tf.Variable(100,dtype=tf.float32) # 바이어스

# 2. 모델 구성 
# y = wx + b
# 행렬 연산이라서 앞 뒤가 바뀌면 값이 달라진다. 
# y = wx + b 원래는 y = xw + b
# y느 hypothesis : 가설
hypothesis = x * w + b

# x,w,b,y만 넣으면 알아서 연산

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 러닝레이트 로스 내려가는 크기 # 미분한다는 것은 그 시점의 가중치를 구하는 것 
# 웨이트 = 웨이트 - 로스 미분 
# 로스 미분 = 방향성
# 웨이트 = 웨이트 - 러닝레이트 x 로스편미분/웨이트 미분
# 경사하강법 
# 미분 그 시점의 기울기 찾기 wx의 미분하면 w 니까! 
train = optimizer.minimize(loss)
# minimize : 최소
# 정사강법 방식으로 옵티마이저를 최적화 시켜준다. 
# 웨이트가 가장 좋은건 연산할 때 판단함
# 로스의 최소값을 뽑는다. 
# w = w - (learning_rate) x (로스 미분)
# model.compile(loss='mse',optimizer='sgd')말 과 똑같다. 
# 훈련 하면 할수록 로스값 준다. 
# 계속 비교하기 때문이다. 


# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())# 초기화
epochs = 2001
for step in range(epochs) : 
    sess.run(train)
    if step %20 == 0 : 
        print(step,sess.run(loss),sess.run(w),sess.run(0)) # verbose

sess.close() 

# 4. 평가 예측
for step in range(epochs) : 
    sess.run(train)
    if step %20 == 0 : 
        print(step,sess.run(loss),sess.run(w),sess.run(0))

