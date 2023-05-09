import tensorflow as tf
print(tf.__version__) #
print('hello world') # hello world
aaa = tf.constant('hello world')
print(aaa)  # Tensor("Const:0", shape=(), dtype=string)

# 상수를 만들기 
# 상수 = 바뀌지 않는 숫자

# h5py -> 텐서플로우 세이브 쪽일 가능성 있음 
# 텐서플로우 연산이 그래프로 됌
# 값을 넣고 그래프로 연산한다. 
# 그래서 결과값이 나오는것이 아니라 모양이 나오는 것이다. 
# 그래서 위에 
# print(aaa)  # Tensor("Const:0", shape=(), dtype=string)
# 이런식으로 도출이 된 것이다. 
# 인풋 -> 세션.런(기계 안에 있는 네모 모양) -> 세션의 결과 빼기 


sess = tf.Session() 
sess = tf.compat.v1.Session() # 세션의 위치기 바뀐 것에 대해서 워닝 때문에 이런식으로 작성 
#WARNING:tensorflow:From C:\study\tf114\tf01_hello.py:20: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
print(sess.run(aaa))
# b'hello world'
# 헬로월드를 이런식으로 출력하고 싶으면 프린트 안에 있는 거처럼 정의 해야한다. 
# import 경로 바꾸기 
