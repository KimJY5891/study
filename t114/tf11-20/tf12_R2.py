import tensorflow as tf

x_train=[1,2,3] # [1]
y_train=[1,2,3] # [2]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10],dtype=tf.float32, name = 'weight')
hypothesis = x *w 

loss = tf.reduce_mean(tf.square(hypothesis-y)) # = (xw-y)^2
 
################ 옵티마이저 ###################
lr = 0.1
grdient = tf.reduce_mean((x*w-y)*x ) #hx
# 바이어스가 빠져있음
# 그라디언트는 로스의 미분값 
# 그라디언트는 w-러닝레이트x미분이랑 연관있다. 
# grdient = tf.reduce_mean((hypothesis-y)*x ) # mean안에 식은 체인룰

descent = w - lr*grdient
# 디센트 갱신되 웨이트
# grdient = 미분하느 부분
update = w.assign(descent)# 새롭게 웨이트로 업데이트
# w = w - lr*grdient

w_history = []
loss_history = []
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 
# x와 y의 피드딕
for step in range(30) :
    _, loss_v,w_v =sess.run([update,loss,w], feed_dict = {x:x_train,y:y_train}) # sess.run은 항상 결과 값을 볼 수 잇다.
    print(step,'\t',loss_v,'\t',w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()


#################### [실습] r2, mae 만들기 ####################
from sklearn.metrics import r2_score,mean_absolute_error

x_test= [4,5,6]
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 
y_pred = x_test * w_v
print(y_pred) 
# y_pred = sess.run(y_pred, feed_dict={x_test: x_test,y:y_train}) 이미 잇는 변수라서 적을 필요 없다. 
# y_pred = sess.run(y_pred,feed_dict ={x:x,update:update} )
r2  = r2_score(y_pred,y_train)
mae = mean_absolute_error(y_pred,y_train)
print('r2 : ',r2)
print('mae : ',mae)
sess.close()import tensorflow as tf

x_train=[1,2,3] # [1]
y_train=[1,2,3] # [2]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10],dtype=tf.float32, name = 'weight')
hypothesis = x *w 

loss = tf.reduce_mean(tf.square(hypothesis-y)) # = (xw-y)^2
 
################ 옵티마이저 ###################
lr = 0.1
grdient = tf.reduce_mean((x*w-y)*x ) #hx
# 바이어스가 빠져있음
# 그라디언트는 로스의 미분값 
# 그라디언트는 w-러닝레이트x미분이랑 연관있다. 
# grdient = tf.reduce_mean((hypothesis-y)*x ) # mean안에 식은 체인룰

descent = w - lr*grdient
# 디센트 갱신되 웨이트
# grdient = 미분하느 부분
update = w.assign(descent)# 새롭게 웨이트로 업데이트
# w = w - lr*grdient

w_history = []
loss_history = []
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 
# x와 y의 피드딕
for step in range(30) :
    _, loss_v,w_v =sess.run([update,loss,w], feed_dict = {x:x_train,y:y_train}) # sess.run은 항상 결과 값을 볼 수 잇다.
    print(step,'\t',loss_v,'\t',w_v)

    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()


#################### [실습] r2, mae 만들기 ####################
from sklearn.metrics import r2_score,mean_absolute_error

x_test= [4,5,6]
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 
y_pred = x_test * w_v
print(y_pred) 
# y_pred = sess.run(y_pred, feed_dict={x_test: x_test,y:y_train}) 이미 잇는 변수라서 적을 필요 없다. 
# y_pred = sess.run(y_pred,feed_dict ={x:x,update:update} )
r2  = r2_score(y_pred,y_train)
mae = mean_absolute_error(y_pred,y_train)
print('r2 : ',r2)
print('mae : ',mae)
sess.close()
