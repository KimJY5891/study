# 컬럼의 갯수가 클래스의 갯수보다 작을 대 디폴트로 돌아가는가? 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, \
    load_digits, load_wine, fetch_covtype
from tensorflow.keras.datasets import cifar100

# 1. 데이터 
datasets_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_digits(return_X_y=True),
    load_wine(return_X_y=True),
    fetch_covtype(return_X_y=True)
]
datasets_name_list = [
    '아이리스',
    '브래스트 캔서',
    '디짓트',
    '와인',
    '패치콥프타입'
]
for i,v  in enumerate(datasets_list) :
    x,y = v
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x,y)
    print(datasets_name_list[i],"x_lda.shape : ",x_lda.shape)
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(datasets_name_list[i],"의 cumsum : ",cumsum)
    
'''
아이리스 x_lda.shape :  (150, 2)
아이리스 의 cumsum :  [0.9912126 1.       ]
브래스트 캔서 x_lda.shape :  (569, 1)
브래스트 캔서 의 cumsum :  [1.]
디짓트 x_lda.shape :  (1797, 9)
디짓트 의 cumsum :  [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
 0.94984789 0.9791736  1.        ]
와인 x_lda.shape :  (178, 2)
와인 의 cumsum :  [0.68747889 1.        ]
패치콥프타입 x_lda.shape :  (581012, 6)
패치콥프타입 의 cumsum :  [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
'''
