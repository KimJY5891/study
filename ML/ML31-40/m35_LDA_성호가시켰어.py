# 컬럼의 갯수가 클래스의 갯수보다 작을 대 디폴트로 돌아가는가? 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from tensorflow.keras.datasets import cifar100

# 1. 데이터 
x,y = load_iris(return_X_y=True)
x,y = load_breast_cancer(return_X_y=True)
x,y = load_digits(return_X_y=True)
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
print(x_train.shape)
pca =PCA(n_components=98)
x_train = pca.fit_transform(x_train)
# print(x.shape)
print(x_train)
x_train = x_train.reshape(50000,32*32*3)
lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis(n_components=3)
x_lda = lda.fit_transform(x_train,y_train)
print(x_lda.shape)

