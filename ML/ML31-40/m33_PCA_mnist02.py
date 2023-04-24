import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import r2_score, accuracy_score
from keras.utils import to_categorical
from sklearn.decomposition import PCA

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 3. Define the original model
model = Sequential()
model.add(Conv2D(7, (2, 2), input_shape=(28, 28, 1)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(Conv2D(10, (2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 4. Compile and train the original model
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=4, batch_size=32, validation_split=0.2, verbose=1)

# 5. Evaluate the original model
original_result = model.evaluate(x_test, y_test)
print('Original Model Result : ', original_result)

# 6. Apply PCA to the MNIST dataset and compare the performance of the original model with PCA models
pca_list = [154, 331, 486, 713]
for i in pca_list:
    # Apply PCA to the data
    pca = PCA(n_components=i)
    x_train_pca = pca.fit_transform(x_train.reshape(x_train.shape[0], -1))
    x_test_pca = pca.transform(x_test.reshape(x_test.shape[0], -1))

    # Create a new model with the same architecture as the original model
    pca_model = Sequential()
    pca_model.add(Dense(14, input_shape=(i,)))
    pca_model.add(Dense(12, activation='relu'))
    pca_model.add(Dense(20))
    pca_model.add(Dense(32, activation='relu'))
    pca_model.add(Dense(10, activation='softmax'))

    # Compile and train the PCA model
    pca_model.compile(loss='categorical_crossentropy', optimizer='adam')
    pca_model.fit(x_train_pca, y_train, epochs=4, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the PCA model
    pca_result = pca_model.evaluate(x_test_pca, y_test)
    print(f'PCA Model ({i} components) Result : {pca_result}')
'''
Original Model loss :  0.07398925721645355
PCA Model (154 components) loss : 0.219069242477417
PCA Model (331 components) loss : 0.21130535006523132
PCA Model (486 components) loss : 0.21130535006523132
PCA Model (713 components) loss : 0.21201929450035095
'''
