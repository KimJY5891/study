import tensorflow as tf
node1 = tf.constant(2.0) 
node2 = tf.constant(3.0)
sess = tf.Session()
# 실습
# 덧셈 node3
node3 = node1 + node2
node3 = tf.add(node1,node2)
print(sess.run(node3)) # 5.0

# 뺄셈 node4
node4 = node1 - node2
node4 = tf.subtract(node1,node2)
print(sess.run(node4)) # -1.0
# 곱셈 node5
node5 = tf.multiply(node1,node2)
print(sess.run(node5)) # 6.0

# 나눗셈  node6
node6 = tf.divide(node1,node2)
print(sess.run(node6)) # 0.6666667

# 만들기
