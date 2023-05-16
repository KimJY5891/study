import tensorflow as tf
tf.compat.v1.set_random_seed(337)

# 1. 데이터

x_data = [[1,2],[2,3], [3,1],[4,3], [5,3],[6,2]] # (6,2)
y_data  = [[0],[0], [0],[1],[1],[1]] # (6,1)

##############################################
#[실습] 시그모이드 빼고 그냥 만들어보기! 
##############################################

x = tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')
# 바이어스는 더하기라 크게 영향을 미치지 않는다. 

# 2. 모델 
hyperthesis = tf.compat.v1.matmul(x,w) + b

# 3-1. 컴파일 
loss = tf.reduce_mean(tf.square(hyperthesis-y))
optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=0.1)
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
from sklearn.metrics import r2_score, mean_absolute_error
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
# y_pred = sess.run(hyperthesis,feed_dict={x:x_data})
x_test = tf.compat.v1.placeholder(tf.float32,shape=[None,2])

y_pred = tf.matmul(x_test,w_val) +b_val
# 넘파이랑 텐서랑 곱했더니 에러가 생김 
# 행렬 곱이라서 멧멀 사용해야함 
y_aaa = sess.run(y_pred,feed_dict={x_test:x_data})  # 텐서 형태 
# y_pred는 텐서 형태 
# y_data는 리스트형태 
# 그래서 다른 데이터 타입이라서 서로 타입을 맞게 만들어야한다. 
# sessrun을 통과하면 넘파이 형태로 변화한다. 

r2 = r2_score(y_data,y_aaa)
print('r2 : ',r2)
mae = mean_absolute_error(y_data,y_aaa)
print('mae : ',mae)
sess.close()
'''
Epoch: 0 Loss: 6.751091 
Epoch: 20 Loss: 6.717108
Epoch: 40 Loss: 6.681952
Epoch: 60 Loss: 6.646135
Epoch: 80 Loss: 6.6097636
Epoch: 100 Loss: 6.5728817
r2 :  -25.28410071316523
mae :  2.5153388579686484

'''

