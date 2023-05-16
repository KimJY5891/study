import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(337)
x_data = [
    [1,2,1,1],
    [2,1,3,2],
    [3,1,3,4],
    [4,1,5,5],
    [1,7,5,5],
    [1,2,5,6],
    [1,6,6,6],
    [1,7,6,7],
]
y_data = [
    [0,0,1], #2
    [0,0,1],
    [0,0,1],
    [0,1,0], #1
    [0,1,0],
    [0,1,0],
    [1,0,0], # 0
    [1,0,0],
]
# 2. 모델 구성 
x= tf.compat.v1.placeholder(tf.float32,shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,3])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,3]))
b= tf.compat.v1.Variable(tf.zeros([3]))

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w)+b)
# softmax => 엔빵이다. 

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))
loss = tf.reduce_mean(tf.reduce_sum(y*tf.log(hypothesis),axis=1)) # 카테고리컬 크로스 엔트로피
train = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=1e-5).minimize(loss)

# 3-2. 훈련 

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 101
for step in range(epochs) :
    _, cost_val, w_val,b_val = sess.run([train,loss,w,b,],feed_dict={x:x_data,y:y_data})
    if step % 20 == 0 :
        print("Epoch:", epochs, "Loss:", cost_val)
        
# 4. 평가 훈련 
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
y_pred = tf.matmul(x_test,w_val)+b_val
# 1,4,5,6,7
y_aaa = sess.run(y_pred,feed_dict={x_test:x_data})
# y_data = tf.compat.v1.convert_to_tensor(y_data) # 텐서 플로우 타입으로 변경해주는 코드 
# y_data = np.array(y_data) # 텐서 플로우 타입으로 변경해주는 코드 
y_aaa = np.argmax(y_aaa,axis=1)
y_data = np.array(y_data)
y_data = np.argmax(y_data,axis=1)
print(type(y_aaa))
print(type(y_data))
print(y_aaa)
print(y_data)
print(y_aaa.shape)
print(y_data.shape)

acc = accuracy_score(y_data,y_aaa) 
print('acc : ',acc)
mae = mean_absolute_error(y_data,y_aaa)
print('mae : ',mae)
sess.close()

'''
Epoch: 101 Loss: -1.3490362
Epoch: 101 Loss: -1.3490362
Epoch: 101 Loss: -1.3490362
Epoch: 101 Loss: -1.3490365
Epoch: 101 Loss: -1.3490365
Epoch: 101 Loss: -1.3490367
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
[0 2 2 2 0 1 1 1]
[2 2 2 1 1 1 0 0]
(8,)
(8,)
acc :  0.375
mae :  0.75
'''

