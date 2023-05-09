import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) # False 
# executing eagerly : 즉시 실행 모드
# 즉시 실행모드(지금까지 해왔던 방식)으로 할껏인가 ? 
# 1.x버전에서는 출력시 false면 즉시 실행모드가 아니라는 것 
# 2.x버전으로 하니까 된다. 
# 2.7.3 -> True
aaa = tf.constant('hello world')
tf.compat.v1.disable_eager_execution() # 즉시 실행모드를 꺼라
print(tf.executing_eagerly()) # False 
tf.compat.v1.enable_eager_execution() # 즉시 실행모드를 꺼라
print(tf.executing_eagerly()) # True 
sess = tf.compat.v1.Session()
print(sess.run(aaa))
# 1버전 : sess.run(aaa) => b'hello world'
# 2버전 -> RuntimeError: The Session graph is empty. Add operations to the graph before calling run().
# => sess.run() 아예 사라짐

# 텐서2.0을 텐서 1.0방식으로 사용
