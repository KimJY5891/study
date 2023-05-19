# pip install keras == 1.2.2
# from tensorflow.keras.datasets import mnist # 1.14버전에 가능하긴함 준비중이었음 
from keras.datasets import mnist
import keras
print(keras.__version__)
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np
import time

# Using TensorFlow backend.
# 텐서플로우 2.-보다 느림 
# 왜냐면 텐서플로우 위에 케라스를 사용하는 것이라 느림 
# 텐서플로우 2.-부터 케라스가 텐서플로우 안에 들어가서 더 빨라짐
# tt.se

# tf.set_random_seed(337) # 1.-버전 가능 
tf.random.set_seed(337) # 2.-버전 가능 

# 1. 데이터 구성

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)-> (60000, 28 * 28) (10000, 28 *28)
print(y_train.shape,y_test.shape) # (60000,) (10000,) -> (60000, 10) (10000, 10)

# 2. 모델 구성 

x = tf.compat.v1.placeholder('float',[None,784])
y = tf.compat.v1.placeholder('float',[None,10])

w1 = tf.get_variable('w1',shape =[784,64]) #  베리어블과 비슷한데 작성방법이 다르다.    
w1 = tf.Variable(tf.random_normal([784,128]))
b1 = tf.Variable(tf.zeros([128]))
layer1 = tf.compat.v1.matmul(x,w1) + b1
dropout1 =tf.compat.v1.nn.dropout(layer1, rate=0.3) # 위에 거 드랍아웃해줌
#model.add(Dropout(0.3))
w2 = tf.Variable(tf.random_normal([128,64]))
b2 = tf.Variable(tf.zeros([64]))
layer2 = tf.nn.relu(tf.compat.v1.matmul(dropout1,w2) + b2)
# activation='relu'

w3= tf.Variable(tf.random_normal([64,32]))
b3 = tf.Variable(tf.zeros([32]))
layer3 = tf.nn.selu( tf.compat.v1.matmul(layer2,w3) + b3)
# activation='selu'
# 셀루 ~ 렐루~ 가능 

w4= tf.Variable(tf.random_normal([32,10]))
b4 = tf.Variable(tf.zeros([10]))
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer3,w4))+b4

# 3-1. 컴파일
# cost = tf.reduce_mean(tf.reduce_sum(y*tf.log(hypothesis),axis=1))
cost = tf.reduce_mean(tf.reduce_sum(y*tf.nn.log_softmax(hypothesis),axis=1))
# cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,lavels=y))
# cost = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

start_time = time.time()
# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs = 100
batch_size = 6000
total_batch = int(len(x_train)/batch_size) # 60000/100 = 600
for epochs in range(epochs) : 
    for i in range(total_batch) :  # 100개씩 600번 돌자 
        start = i *batch_size       # 0, 100,200 ... 59900
        end  = start + batch_size   # 100,200,300 ... 60000
        # 배치 사이즈는 데이터를 조각조각내서 훈련시키는 것이다. ㅣ
        # 토치는 데이터를 미리 잘라서 넣는다. 
        # 첫 번째 배치 x_train[:100], y_train[:100]
        # 배치 x_train[start:end], y_train[start:end]
        cost_val, _,w_val,b_val = sess.run([cost,train,w4,b4],feed_dict={x:x_train[start:end],y:y_train[start:end]})
        # cost_val 600개 나오는데, 그렇게 600개 평균하면 된다. 
        avg_cost = cost_val/total_batch
    print("Epoch : ", epochs+1, "Loss : {:.9f}".format(avg_cost))
end_time = time.time()
    
print('훈련끝')        


# 4. 평가 훈련 
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
y_pred = sess.run(hypothesis,feed_dict= {x:x_test})
y_pred_arg = sess.run(tf.argmax(y_pred,axis=1))
print(type(y_test))
print(type(y_pred_arg))

acc = accuracy_score(y_test,y_pred_arg) 
print('acc : ',acc)
mae = mean_absolute_error(y_test,y_pred_arg)
print('mae : ',mae)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 캐스팅 자료 형을 바꿔줘라 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32)) # 액큐러시의 수식 

a=sess.run([accuracy],feed_dict={x:x_test,y:y_test})
print("accuracy : ",acc)
print("tf",tf.__version__,'걸린 시간 : ',round(end_time-start_time,2))

sess.close()
