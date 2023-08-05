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


model_name_list = ['001','008','032','064','154','331','486','713']
model_num_list = [1,8,32,64,154,331,486,713]
model_list = []

for i,v in enumerate(model_num_list):
    print(model_name_list[i])
    model_name = 'model_' + str(model_name_list[i])
    locals()[model_name] = autoencoder(hidden_layer_size= v )
    # 틀린 거 :  locals()['model'+str(model_name_list[i])] = autoencoder(hidden_layer_size = v)
    model_list.append(locals()[model_name])
    # 틀린 거 : model_list.append(locals()['model_'+str(model_name_list[i])]) 
    print(model_list)
    
# 3. 컴파일, 훈련
for i,v in enumerate(model_list) : 
    print(f'========================== node {model_name_list[i]}개 시작 ============================')
    v.compile(optimizer = 'adam', loss = 'mse', )
    v.fit(x_train_noised, x_train, epochs = 10, batch_size = 128, validation_split = 0.2)

decoded_imgs_list = [ ]
# 4. 평가, 예측
for i,v in enumerate(model_num_list):
    print(model_name_list[i])
    decoded_imgs_name = 'decoded_imgs_' + str(model_name_list[i])
    decoded_imgs_list.append(locals()[decoded_imgs_name])
    print(decoded_imgs_list)
    
for i,v in enumerate(decoded_imgs_list) : 
    v = model_list[i].predict(x_test_noised)
    

############################# 아무 생각 없이 타자 연습 ######################
import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(9,5,figsize=(15,15))
random_images = random.sample(range(decoded_imgs_list[i].shape[0]),5)
outputs = [x_test] 
outputs = outputs + decoded_imgs_list
for row_num, row in enumerate(axes) : 
    for col_num, ax in enumerate(row) : 
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28),cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()
