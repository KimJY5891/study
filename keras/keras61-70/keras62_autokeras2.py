#자동화 머신러닝
import autokeras as ak

print(ak.__version__)

import tensorflow as tf
import time
path = './_save/autokeras/'

#1. 데이터 
(x_train, y_train), (x_test, y_test) = \
                            tf.keras.datasets.mnist.load_data()

#2. 모델 
model = ak.ImageClassifier(
    overwrite = False,
    max_trials=2
)
model.summary()
# model.load(path+'')
# 로드 모델 가능 
#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs = 10, validation_split = 0.15)
end = time.time()

#4. 평가, 예측 
y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)

# 
