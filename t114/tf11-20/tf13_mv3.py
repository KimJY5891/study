import tensorflow as tf
tf.compat.v1.set_random_seed(337)
x_data = [
    [73,51,65],
    [92,98,11],
    [89,31,33],
    [99,33,100],
    [17,66,79],
          ] # (5,3)

y_data = [
    [152],
    [185],
    [180],
    [205],
    [142],
          ] # 5,1

x = tf.compat.v1.placeholder(tf.float32, shape=[None,3])
# 지정하지 않으면 알아서 해주지만, 안전빵으로 우리가 정해줘야한다. 
# 행의 갯수는 바뀔 수도 있어서 None으로 표시해 준다. 
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),name = 'weight')
# ([]) = 쉐이프가 없다는 말 , 알아서 작성해주셈요 .
# ([3,1])
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name = 'bias')
# 2. 모델 
#  hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
# hypothesis = x * w + b
# 완전한 행렬 연산이 아니기 때문에 오류 날 수도 잇다. 
hypothesis = tf.compat.v1.matmul(x,w) + b
# 이게 더 안전하다.  오류 날 경우 그냥 이거 사용하는게 좋다. 
# x.shape(5,3)
# y.shape(5,1)
# hypothesis = x * w +b 인데 ,
# x 는 5행 3열 , 
# (5,3) *w+b = (5,1)
# (5,3)(=x)*(?,?)(=w) = (5,1 ) 답 : 3,1 
# 3-1. 컴파일
########################## [실습] 만들어 보기 ##########################

loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 
epochs = 101
for  step in range(23) : 
    cost_val, _ = sess.run([loss,train],feed_dict={x:x_data,y:y_data})
    if step % 20 == 0 :
        print("Epoch:", step, "Loss:", cost_val)

# 4. 평가, 예측3
# r2, mse로 평가 

from sklearn.metrics import r2_score, mean_absolute_error

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
y_pred = sess.run(hypothesis, feed_dict={x: x_data})
r2 = r2_score(y_data, y_pred)
mse = mean_absolute_error(y_data, y_pred)
print('r2:', r2)
print('mse:', mse)
sess.close()


'''
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
y_pred = x_test * w_v
r2 = r2_score(y_train,y_pred)
mse = mean_absolute_error(y_train,y_pred)
print('r2 : ',r2)
print('mse : ',mse)
sess.close()
'''
