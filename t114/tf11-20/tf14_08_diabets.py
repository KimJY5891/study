
import tensorflow as tf
tf.compat.v1.set_random_seed(337)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np

# 1. 데이터
x,y=load_diabetes(return_X_y=True)
print(x.shape,y.shape)
print(y[:10]) # [151.  75. 141. 206. 135.  97. 138.  63. 110. 310.] 이런식으로 보이지만 실질적으로 회귀 데이터

y = y.reshape(-1,1) # (442,1)
'''
 x(442,10) * w(?,?)+b = u(442,1)
w의 답 : (10,1)
'''

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size = 0.8,
    shuffle = True,
    random_state= 337,
)
print(x_train.shape,y_train.shape) # (353, 10) (353, 1)
print(x_test.shape,y_test.shape) # (89, 10) (89, 1)

xp = tf.compat.v1.placeholder(tf.float32,shape=[None,10])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),name='weights')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')

# 2. 모델
hyperthesis = tf.compat.v1.matmul(x,w) + b

# 3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hyperthesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
# 옵티마이저 아담 구현하기 
# 아담 또한 
# 그라디언트 디센트 옵티마이저에
train  = optimizer.minimize(loss)

# 3-2. 훈련 
sess =tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for epochs in range(epochs) : 
    cost_val, _ = sess.run([loss,train],feed_dict = {x:x_train,y:y_train})
    if epochs % 20 ==0:
        print("Epoch:", epochs, "Loss:", cost_val)

# 점심 때 만들어보기 

# 4. 평가, 예측
# r2, mse로 평가

from sklearn.metrics import r2_score, mean_absolute_error

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
y_pred = sess.run(hyperthesis,feed_dict= {x:x_test})
r2 = r2_score(y_test, y_pred)
mse = mean_absolute_error(y_test, y_pred)
print('r2:', r2)
print('mse:', mse)
sess.close()
