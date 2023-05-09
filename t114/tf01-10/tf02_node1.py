import tensorflow as tf

node1 = tf.constant(3.0,tf.float32) # 3.0이며 float 형태이다.
node2 = tf.constant(4.0) # 4.0이며 float  형태이다. 
# 이름이 노드인 이유 노드 연산 방식이라서 
# 노드1과 노드2가 있고 이것을 연산하는 방식을 만들 것이다. 노드 3(연산 방식)
# 이 방식 하나가 텐서 머신이다. 
# 내가 뽑고 싶은 부분을 아웃풋하면 된다. 
# 그 아웃풋을 결정하는것이 sess.run()

# node3 = node1+ node2
node3 = tf.add(node1,node2) # 위와 결과는 동일 하다.
print(node1) # Tensor("Const:0", -> 상수 0번째 
# shape=(), dtype=float32)
print(node2) # Tensor("Const_1:0", -> 상수1번째 
# shape=(), dtype=float32)
print(node3) # Tensor("add:0", => 더하기 연산이다 라는 의미 
# shape=(), dtype=float32) -> 그래프의 모양

sess = tf.compat.v1.Session()
print(sess.run(node1)) # 3.0
print(sess.run(node2)) # 4.0
print(sess.run(node3)) # 7.0



