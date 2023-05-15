import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error

# 1. 데이터
x,y = load_iris(return_X_y=True)
print(x.shape)
print(y.shape)
y =y.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size = 0.8,random_state=337,shuffle=True,stratify=y
)

xp = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,1]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name = 'bias')

# 2. 모델 구성
# hyperthesis  = tf.compat.v1.matmul(xp,w) + b
hypothesis  =  tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp,w) + b)

# 3-1. 컴파일 
# loss = tf.reduce_mean(tf.square(hyperthesis-yp)) # mse
# loss = tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
loss = tf.reduce_mean(y * tf.log_sigmoid(hypothesis) + (1 - y) * tf.log_sigmoid(1 - hypothesis))
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yp, logits=hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)
# 금요일 휴강 다음주부터 텐서2나 파이토치

# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(loss,feed_dict={xp:x_train,yp:y_train}))
epochs = 100
for epochs in range(epochs) :
    _ ,cost_val, w_val, b_val = sess.run([train,loss,w,b],feed_dict = {xp:x_train,yp:y_train})  
    if epochs % 20 == 0 : 
        print("Epoch:", epochs, "Loss:", cost_val)
    sess.run(tf.compat.v1.global_variables_initializer())


# 4. 평가 훈련 

from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y_pred = tf.compat.v1.sigmoid(tf.matmul(x_test,w_val)+b_val)
y_pred = tf.cast(y_pred>0.5,dtype=tf.float32)
y_aaa = sess.run(y_pred,feed_dict={xp:x_test,yp:y_test})

acc = accuracy_score(y_test,y_aaa)
print('acc : ',acc)
mae = mean_absolute_error(y_test,y_aaa)
print('mae : ',mae)
sess.close()
