# 멀티 배리어블 
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
# 플레이스 홀더 만들기( 변수 )
# 1. 데이터
        #  1열 2열 3열 4열 5열 
x1_data = [73.,93.,89.,96.,73.]      # 국어
x2_data = [80.,88.,91.,98.,66.]      # 영어
x3_data = [75.,93.,90.,100.,70.]     # 수학
y1_data = [152.,185.,180.,196.,142.] # 환산점수

# [실습] 만들어보기

x1= tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y= tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))

# 2. 모델 

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis-y))
# 옵티마이저
lr = 0.1
# grdient = tf.reduce_mean(((x1 * w1 + x2 * w2 + x3 * w3 + b)-y)*(x1+x2+x3))
grdient = tf.reduce_mean(((x1 * w1 + x2 * w2 + x3 * w3  + b)-y)*(x1+x2+x3))


descent01 = w1 -lr*grdient
descent02 = w2 -lr*grdient
descent03 = w3 -lr*grdient
descent_b = b - lr * tf.reduce_mean((hypothesis - y))

# 업데이트된 가중치와 바이어스 적용
update_w1 = w1.assign(descent01)
update_w2 = w2.assign(descent01)
update_w3 = w3.assign(descent03)
update_b = b.assign(descent_b)

# 3-2. 훈 련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
epochs = 101
w1_history = []
w2_history = []
w3_history = []
b_history = []
loss_history = []
for step in range(epochs) : 
    _, _, _, loss_val, w1_val,w2_val,w3_val, b_val  = sess.run([update_w1,update_w2,update_w3,loss,w1,w2,w3,b], feed_dict={x:x_train,y:y_train})

    w1_history.append(w1_val)
    w2_history.append(w2_val)
    w3_history.append(w3_val)
    b_history.append(b_val)
    loss_history.append(loss_val)
sess.close()

#################### [실습] r2, mae 만들기 ####################
from sklearn.metrics import r2_score,mean_absolute_error

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
y_pred = x_test * w1_val
print(y_pred) 
# y_pred = sess.run(y_pred, feed_dict={x_test: x_test,y:y_train}) 이미 잇는 변수라서 적을 필요 없다. 
# y_pred = sess.run(y_pred,feed_dict ={x:x,update:update} )
r2  = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('r2 : ',r2)
print('mae : ',mae)
sess.close()
'''
from chatgpt
따로 계산해서 사용하는 방법 
# 2. 모델
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
lr = 0.1

# 편미분 계산
grdient01 = tf.reduce_mean(((x1 * w1 + b) - y) * (x1 + x2 + x3))
grdient02 = tf.reduce_mean(((x2 * w2 + b) - y) * (x1 + x2 + x3))
grdient03 = tf.reduce_mean(((x3 * w3 + b) - y) * (x1 + x2 + x3))

# 경사하강법 적용
update_w1 = w1 - lr * grdient01
update_w2 = w2 - lr * grdient02
update_w3 = w3 - lr * grdient03
update_b = b - lr * tf.reduce_mean((hypothesis - y))

# 업데이트된 가중치와 바이어스 적용
update_w1 = tf.update(w1, update_w1)
update_w2 = tf.update(w2, update_w2)
update_w3 = tf.update(w3, update_w3)
update_b = tf.update(b, update_b)

# 텐서플로 세션 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 경사하강법 반복 학습
for step in range(21):
    _, loss_v, w1_v, w2_v, w3_v, b_v = sess.run([update_w1, update_w2, update_w3, update_b, loss, w1, w2, w3, b], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    print(step, '', loss_v, '', w1_v, '', w2_v, '', w3_v, '', b_v)

모두 합쳐서 사용하는 방법
# 2. 모델
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
lr = 0.1

# 편미분 계산
gradient = tf.reduce_mean(((hypothesis - y) * (x1 + x2 + x3)))

# 경사하강법 적용
update_w1 = w1 - lr * gradient
update_w2 = w2 - lr * gradient
update_w3 = w3 - lr * gradient
update_b = b - lr * tf.reduce_mean((hypothesis - y))

# 업데이트된 가중치와 바이어스 적용
update_w1 = tf.update(w1, update_w1)
update_w2 = tf.update(w2, update_w2)
update_w3 = tf.update(w3, update_w3)
update_b = tf.update(b, update_b)

# 텐서플로 세션 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer

'''
