import tensorflow as tf
tf.compat.v1.set_random_seed(337)

# 1. 데이터

x_data = [
    [1,2],
    [2,3],
    [3,1],
    [4,3],
    [5,3],
    [6,2]
    ] # (6,2)
y_data  = [[0],
           [0],
           [0],
           [1],
           [1],
           [1],] # (6,1)

##############################################
#[실습] 시그모이드 빼고 그냥 만들어보기! 
##############################################

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')
# 바이어스는 더하기라 크게 영향을 미치지 않는다. 

# 2. 모델 

hyperthesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w) + b)
# sigmoid(hyperthesis)
# 영과 1사이에 있음 


# 3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hyperthesis-y)) # mse
loss = tf.reduce_mean(y*tf.log(hyperthesis-y) + (1-y)*tf.log(1-hyperthesis)) # 바이너리컬 센트로피 
# 무조건 둘 중 하나만 움직인다. 
# y가 0일 경우 앞이 0
# y가 1일 경우 뒤가 0

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train =optimizer.minimize(loss)

# 3-2. 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for epochs in range(epochs) : 
    cost_val, _, w_val,b_val  = sess.run([loss,train,w,b],feed_dict={x:x_data,y:y_data})
    if epochs % 20 == 0 :
        print("Epoch:", epochs, "Loss:", cost_val)
        
# 4. 평가 훈련         
from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y_pred = tf.sigmoid(tf.matmul(x_test,w_val) +b_val)
# w_val은 넘파이 
y_pred = tf.cast(y_pred>0.5,dtype=tf.float32) # 라운드 가능하다. 
# cast  :  0.5이상이면  자료 형 정수로 바꿔달라 
y_aaa = sess.run(y_pred,feed_dict={x_test:x_data})  # 텐서 형태 
# x_test,x_data는 리스트 
# y_aaa는 텐서 형태 

acc = accuracy_score(y,y_aaa)
print('acc : ',acc)
mae = mean_absolute_error(y,y_aaa)
print('mae : ',mae)
sess.close()


