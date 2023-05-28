import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(337)
x_data = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
],dtype=np.float32)
y_data = np.array([
    [0],[1],[1],[0]
],dtype=np.float32)

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')

# 2. 모델 
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b)

# 3-1. 컴파일 
cost = tf.reduce_mean(y*tf.log_sigmoid(hypothesis) + (1-y)*tf.log_sigmoid(1-hypothesis)) # 바이너리컬 센트로피 
# 1이면 true로 반환,0이면 false로 반환 
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 캐스팅 자료 형을 바꿔줘라 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32)) # 액큐러시의 수식 
# 그래서 나온  1,1,1,1 predicted값 에 대해서 equal()에서 true면 1,아니면 0으로 반환한다. 
# 그래서 평균으로 나눠서 0과1사이로 나온다. 

# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 5001
for epochs in range(epochs) : 
    cost_val, _, = sess.run([cost,train],feed_dict={x:x_data,y:y_data})
    # cost_val, _, w_val,b_val  = sess.run([loss,train,w,b],feed_dict={x:x_data,y:y_data})
    if epochs % 200 == 0 : # 200번 마다 한 번씩 보느 ㄴ것 
        print("Epoch:", epochs, "Loss:", cost_val)
h,p,a=sess.run([hypothesis,predicted,accuracy],feed_dict={x:x_data,y:y_data})
print("예측값 : ",h,"\n 원래값 : ",p,"\n accuracy : ",a)


'''
예측값 :  
[9.9997437e-01]
 [1.0000000e+00]
 [3.2147331e-05]
 [9.9998915e-01]
 원래값 :  
 [1.]
 [1.]
 [0.]
 [1.]
 accuracy :  0.25
'''

'''
# 4. 평가 훈련         
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y_pred = tf.sigmoid(tf.matmul(x_test,w_val) +b_val)
y_pred = tf.cast(y_pred>0.5,dtype=tf.float32)
y_aaa = sess.run(y_pred,feed_dict={x_test:x_data})  

acc = accuracy_score(y_data,y_aaa)
print('acc : ',acc)
mae = mean_absolute_error(y_data,y_aaa)
print('mae : ',mae)
sess.close()
'''
