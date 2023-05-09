# 변수,상수, 플레이

import tensorflow as tf
print(tf.__version__)  # 1.14.0
print(tf.executing_eagerly())  # False
# 즉시실행모드 
tf.compat.v1.disable_eager_execution() # 즉시 실행모드 꺼
print(tf.executing_eagerly())  # False
# 실행모드 끄면 1버전것 사용 가능
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)
sess = tf.compat.v1.Session()
print()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
# placeholder : 공간고정? 
# 어떤 공간에 값을 받을 준비를 하는것 
# 빈 공간 
# 노드 상수 말고 빈자리로 일단 만들어주는것 , 변수와 상수의 중간정도의 상태

add_node = a + b
print(sess.run(add_node,feed_dict={a:3,b:4.5})) # 7.5
#빈자리에 숫자 넣어주기
print(sess.run(add_node,feed_dict={a:[1,3],b:[y_pred2,4]})) # [2. 6.]
# 텐서플로우는 행렬 연산 

add_and_triple = add_node * 3
print(add_and_triple) # Tensor("mul:0", dtype=float32) 그래프로 나온다. 

print(sess.run(add_and_triple,feed_dict={a:7,b:3})) # 30
# 꺼꾸로 생각해서 계산하는 것이 가장 좋다. 


