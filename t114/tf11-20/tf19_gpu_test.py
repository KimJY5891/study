import tensorflow as tf

# tf.compat.v1.disable_eager_execution() # 즉시모드 끔 -> 1.x버전
# tf.compat.v1.enable_eager_execution() # 즉시모드 킴 -> 2.x버전
print(tf.__version__)
print(tf.executing_eagerly())

# gpu 버전 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus : 
    try :
        tf.config.experimental.set_visible_devices(gpus[0],'GPU')
        print(gpus[0])
    except RuntimeError as e :
        print(e)
else :
    print('gpu 없다!')
# 텐서 2.7.4gpu버전 실행결과
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
