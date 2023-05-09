import tensorflow as tf

sess = tf.compat.v1.Session()
x = tf.Variable([2],dtype=tf.float32) 
y = tf.Variable([3],dtype=tf.float32)
# 배리어블을 사용할 때, 변수를 초기화를 해줘야한다. 
# 배리어블은 변수 사용전에 변수 초기화를 해줘야한다. 
init = tf.compat.v1.global_variables_initializer() # 변수 초기화 
sess.run(init)
print(sess.run(init)) # print()
print(sess.run(x+y)) 
# self._traceback = tf_stack.extract_stack()
