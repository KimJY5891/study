from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
# Convolution2D = conv2D


#2. 모델 구성
model = Sequential()                     #(n,3)
model.add(Dense(10, input_shape=(3,)))   #(batch_size, input_dim)
model.add(Dense(units=15))               #출력(batch_size, units)
model.summary()    
#units = 노드


#비교 정리


# tf.keras.layers.Dense(
#     units, # 아웃풋 노드의 갯수
#     #필터와 같은 내겸
#     activation=None,
#     use_bias=True,
#     kernel_initializer="glorot_uniform",# 대문자는 댑부분 클랫
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )


# >>> # Create a `Sequential` model and add a Dense layer as the first layer.
# >>> model = tf.keras.models.Sequential()
# >>> model.add(tf.keras.Input(shape=(16,)))# 덴스도 다차원 가능하다.
# >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
# >>> # Now the model will take as input arrays of shape (None, 16)
# >>> # and output arrays of shape (None, 32).
# >>> # Note that after the first layer, you don't need to specify
# >>> # the size of the input anymore:
# >>> model.add(tf.keras.layers.Dense(32))
# >>> model.output_shape
# (None, 32)
