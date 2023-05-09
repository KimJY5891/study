import tensorflow as tf

##### 현재 버전이 1.0이면 그냥 출력 
##### 현재 버전이 2.0아면 즉시 실행 모드 끄고 출력 

'''
if float(tf.__version__.split('.')[0]) >= 1.0 and float(tf.__version__.split('.')[0]) <2:
    print(tf.__version__)
    print(tf.executing_eagerly()) # False 
    aaa = tf.constant('hello world')
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))
    
elif float(tf.__version__.split('.')[0]) >= 2 : 
    print(tf.__version__)
    print(tf.executing_eagerly()) # False 

    aaa = tf.constant('hello world')
    tf.compat.v1.disable_eager_execution() # 즉시 실행모드를 꺼라
    print(tf.executing_eagerly()) # True 
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))
'''


# 혹은 즉시실행모드 켜진거인지 아닌걸로 조건을 만들어서 작성
if print(tf.executing_eagerly()) == False :
    print(tf.__version__)
    print(tf.executing_eagerly()) # False 
    aaa = tf.constant('hello world')
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))
    
else : 
    print(tf.__version__)
    print(tf.executing_eagerly()) # True 
    tf.compat.v1.disable_eager_execution() # 즉시 실행모드를 꺼라
    print(tf.executing_eagerly()) # True 
    aaa = tf.constant('hello world')
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))


'''
    
if not tf.executing_eagerly():
    print(tf.__version__)
    print(tf.executing_eagerly()) # False 
    aaa = tf.constant('hello world')
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))
    
else:
    print(tf.__version__)
    print(tf.executing_eagerly()) # True 
    tf.compat.v1.disable_eager_execution() # 즉시 실행모드를 꺼라
    aaa = tf.constant('hello world')
    print(tf.executing_eagerly()) # False 
    sess = tf.compat.v1.Session()
    print(sess.run(aaa))
'''
    
