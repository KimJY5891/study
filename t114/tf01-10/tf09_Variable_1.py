import tensorflow as tf
tf.compat.v1.set_random_seed(123)
# tf.set_random_seed(123) : 위와 동일 그냥 경고 뜸

변수  = tf.compat.v1.Variable(tf.random_normal([2
                                              ]),name='weight')
# tf.random_normal([1]) 쉐이프 1 숫자 하나
print(변수) # 변수에 들어가는건 숫자 
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수) # aaa :  [-1.5080816   0.26086742]
print('aaa : ',aaa) # 
sess.close() 
# 닫았으니 다시 열어야함

# 초기화 두 번째

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb =  변수.eval(session=sess) # 텐서플로우 데이터 형인 '변수'를 파이썬에서 볼 수 있는 놈으로 바꿔줘
print(bbb) # [-1.5080816   0.26086742]
#  eval => 파이썬 데이터 형태로 바꿔준다. 
# 그러면 sess.run 하지 않아도 된다. 
# 그래도 sess을 사용했기 때문에 클로즈 해줘야한다. 
sess.close() 

# 초기화 세 번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval() # InteractiveSession 으로 되어 잇다면 sess 안 사용해도 된다. 
print(ccc)  # [-1.5080816   0.26086742]
sess.close()

# 사용하고 싶은 거 사용하면 된다. 
# 3년 전에는 리뉴얼 했지만 ,지금은 케라스 사용 중이다. 
# 옛날 사용하는 회사 있을 지도 있다. 
