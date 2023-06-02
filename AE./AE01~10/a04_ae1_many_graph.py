import numpy as np
from tensorflow.keras.datasets import mnist


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.5, size = x_train.shape) #정규분포 형식의 임의의 값
x_test_noised = x_test + np.random.normal(0, 0.5, size = x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)

print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))


x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1) #최저값 0 최고값 1로 고정시키는 함수

print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (784,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

model_001 = autoencoder(hidden_layer_size=1) #PCA 100% 성능
model_008 = autoencoder(hidden_layer_size=8) #PCA 100% 성능
model_032 = autoencoder(hidden_layer_size=32) #PCA 100% 성능
model_064 = autoencoder(hidden_layer_size=64) #PCA 100% 성능
model_154 = autoencoder(hidden_layer_size=154) #PCA 100% 성능
model_331 = autoencoder(hidden_layer_size=331) #PCA 99% 성능
model_486 = autoencoder(hidden_layer_size=486) #PCA 99.9% 성능
model_713 = autoencoder(hidden_layer_size=713) #PCA 100% 성능

# 3. 컴파일, 훈련

# for i,v in enumerate
print('========================== node 1개 시작 ============================')
model_001.compile(optimizer = 'adam', loss = 'mse', )
model_001.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)
print('========================== node 8개 시작 ============================')
model_008.compile(optimizer = 'adam', loss = 'mse', )
model_008.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)
print('========================== node 32개 시작 ============================')
model_032.compile(optimizer = 'adam', loss = 'mse', )
model_032.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)
print('========================== node 64개 시작 ============================')
model_064.compile(optimizer = 'adam', loss = 'mse', )
model_064.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)
print('========================== node 154개 시작 ============================')
model_154.compile(optimizer = 'adam', loss = 'mse', )
model_154.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)
print('========================== node 331개 시작 ============================')
model_331.compile(optimizer = 'adam', loss = 'mse', )
model_331.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)
print('========================== node 486개 시작 ============================')
model_486.compile(optimizer = 'adam', loss = 'mse', )
model_486.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)
print('========================== node 713개 시작 ============================')
model_713.compile(optimizer = 'adam', loss = 'mse', )
model_713.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)

# 4. 평가, 예측

# decoded_imgs_ = np.round(autoencoder.predict(x_test))
decoded_imgs_001 = model_001.predict(x_test_noised)
decoded_imgs_008 = model_008.predict(x_test_noised)
decoded_imgs_032 = model_032.predict(x_test_noised)
decoded_imgs_064 = model_064.predict(x_test_noised)
decoded_imgs_154 = model_154.predict(x_test_noised)
decoded_imgs_331 = model_331.predict(x_test_noised)
decoded_imgs_486 = model_486.predict(x_test_noised)
decoded_imgs_713 = model_713.predict(x_test_noised)

############################# 아무 생각 없이 타자 연습 ######################
import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(7,5,figsize=(15,15))
random_images = random.sample(range(decoded_imgs_001.shape[0]),5)
outputs = [x_test,decoded_imgs_001,decoded_imgs_008,decoded_imgs_032,decoded_imgs_064,decoded_imgs_154,decoded_imgs_331,
           decoded_imgs_486,decoded_imgs_713
           ]
for row_num, row in enumerate(axes) : 
        for col_num, ax in enumerate(row) : 
            ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28),cmap='gray')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
plt.show()
