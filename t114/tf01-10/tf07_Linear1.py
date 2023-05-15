import tensorflow as tf

tf.set_random_seed(337)

# 1. 데이터
x = [1,2,3]
y = [1,2,3]
w = tf.Variable(111,dtype=tf.float32) # 가중치
b = tf.Variable(100,dtype=tf.float32) # 바이어스

# 2. 모델 구성 
# y = wx + b
# 텐서플로 연산방식이 행렬 연산이라서 앞 뒤가 바뀌면 값이 달라진다. 
# y = wx + b 원래는 y = xw + b

hypothesis = x * w + b
# hypothesis : 가설
# x,w,b,y만 넣으면 알아서 연산

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # 러닝레이트 로스 내려가는 크기 # 미분한다는 것은 그 시점의 가중치를 구하는 것 
# 웨이트 = 웨이트 - ( 러닝레이트  * (로스 미분 / 가중치 미분))
# 로스 미분 = 방향성
# 웨이트 = 웨이트 - 러닝레이트 x 로스편미분/웨이트 미분
# 경사하강법 
# 미분 그 시점의 기울기 찾기 xw의 미분하면 w 니까! 
train = optimizer.minimize(loss)
# minimize : 최소
# 경사하강법 방식으로 옵티마이져를 최적화하여 loss의 최소값을 뽑는다
# 웨이트가 가장 좋은건 연산할 때 판단함
# w = w - (learning_rate) x (로스 미분)
# model.compile(loss='mse',optimizer='sgd')말 과 똑같다. 
# 훈련 하면 할수록 로스값 준다. 
# 계속 비교하기 때문이다. 


# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) #세션을 열면 항상 초기화부터 실행
epochs = 2001
for step in range(epochs) : 
    sess.run(train)
    if step %100 == 0 :  #100으로 나눈 나머지가 0일때
        print(step,sess.run(loss),sess.run(w),sess.run(b)) # verbose

sess.close()   #세션 종료. 생성한 메모리 terminate

# 미분 = 그 지점의 변화량 구하기(5.12 시점)
