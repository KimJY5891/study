import tensorflow as tf

x_train=[1,2,3] # [1]
y_train=[1,2,3] # [2]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10],dtype=tf.float32, name = 'weight')


# 2. 모델 
hypothesis = x *w 

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))
 
################ 옵티마이저 ###################
lr = 0.1
grdient = tf.reduce_mean((x*w-y)*x)
# 그라디언트는 w-러닝레이트x미분이랑 연관있다. 
# grdient = tf.reduce_mean((hypothesis-y)*x ) # mean안에 식은 체인룰

descent = w - lr*grdient
# grdient = 미분하느 부분
update = w.assign(descent)# 새롭게 웨이트로 업데이트
# w = w - lr*grdient

w_history = []
loss_history = []
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 
# x와 y의 피드딕
for step in range(21) :
    _, loss_v,w_v =sess.run([update,loss,w], feed_dict = {x:x_train,y:y_train}) # sess.run은 항상 결과 값을 볼 수 잇다.
    print(step,'\t',loss_v,'\t',w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()
print('============== ')
print( w_history)
print('============== ')
print(loss_history)
