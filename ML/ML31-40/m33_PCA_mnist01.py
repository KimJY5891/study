import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train,_),(x_test,_) = mnist.load_data()
# y는 안 뽑고 x만 하고 싶을 때 뽑고 싶지 않는 자리에 _만 작성해주면 된다.
# (x_train,y_trian),(x_test,y_test) = mnist.load_data()

# print(_.shape) # 10000
# 전체가 컬럼이 28x28개라고 할때 거의 0이다.
# 0 0 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 0 0 0 0 0 
# 0 0 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 0 0 0 0 0 
# 0 0 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 0 0 0 0 0 
# 0 0 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 0 0 0 0 0 
# 0 0 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 0 0 0 0 0 
# 0 0 0 0 0 0 0 0 0 0 0 0 .... 0 0 0 0 0 0 0 0 0 
# 양끝 열을 0이라서 압축하는게 좋다. 
# pca로 압축
# 
# x = np.concatenate((x_train,x_test),axis=0)
x = np.append(x_train,x_test,axis=0)
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
print(x.shape) # (70000, 784)

################### 실습 ###################
# pca를 통해 0.95 이상인 n_components는 몇개?
# 0.95 몇개
# 0.99 몇개
# 0.999 몇 개 
# 1.0 몇 개 

pca = PCA(n_components=784)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
print(cumsum)
print(np.argmax(cumsum >= 0.95)+1) #154
print(np.argmax(cumsum >= 0.99)+1) # 331
print(np.argmax(cumsum >= 0.999)+1) # 486
print(np.argmax(cumsum >= 1.0)+1) #713
# 처음이 0이니까 +1 해줘야함 


'''
for i in range(x.shape[1]-1):
    (x_train,_),(x_test,_) = mnist.load_data()
    x = np.append(x_train,x_test,axis=0)
    x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) 
    pca = PCA(n_components=(i+1))
    x = pca.fit_transform(x)
    pca_EVR = pca.explained_variance_ratio_
    # explained_variance_ratio_ = 설명가능한 변화율
    print(i+1,"번째 : ",sum(pca_EVR))
    if sum(pca_EVR) >= 0.95 : 
        pca_evr_for_list_95 = []
        pca_evr_for_list_95.append(pca_EVR)
        if sum(pca_EVR) >= 0.95 and sum(pca_EVR) < 0.99 : 
            pca_evr_for_list_95_99 = []
            pca_evr_for_list_95_99.append(pca_EVR)
            if sum(pca_EVR) >= 0.99 and sum(pca_EVR) < 0.999 : 
                pca_evr_for_list_99_999 = []
                pca_evr_for_list_99_999.append(pca_EVR)
                if sum(pca_EVR) > 1. : 
                    pca_evr_for_list_1 = []
                    pca_evr_for_list_1.append(pca_EVR)
print('0.95이상의 갯수 : ',len(pca_evr_for_list_95))   
print('0.95이상이고 0.99미만의 갯수 : ',len(pca_evr_for_list_95_99))   
print('0.99이상이고 0.999미만의 갯수 : ',len(pca_evr_for_list_99_999))   
print('1.의 갯수 : ',len(pca_evr_for_list_1))   

'''
