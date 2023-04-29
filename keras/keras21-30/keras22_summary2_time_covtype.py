from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from tensorflow.keras.losses import sparse_categorical_crossentropy

datasets= fetch_covtype()

x=datasets['data']
y=datasets.target
print(datasets)
print(datasets.DESCR)
print("datasets.feature_names :",datasets.feature_names)
print(x.shape,y.shape) #(581012, 54) (581012,)
print('y의 라벨 값 :',np.unique(y)) #  [1 2 3 4 5 6 7]
# encoder = OneHotEncoder(sparse=False)
# y = encoder.fit_transform(y.reshape(-1, 1))
"""
y.reshape(-1, 1)은 y 배열을 2차원 배열로 변환하는데,
첫 번째 차원은 -1로 설정되어 있습니다. -1을 지정하면 해당 차원은 자동으로 계산되어 지정됩니다.
이 경우에는 y 배열의 원래 shape에서 첫 번째 차원의 크기와 일치하도록 설정됩니다.
이렇게 2차원 배열로 변환하는 이유는 OneHotEncoder에서 인자로 넘겨줄 데이터는 2차원 배열이어야 하기 때문입니다.
1차원 배열이라면 reshape 함수를 사용하여 2차원 배열로 변환해 주어야 합니다.
"""
print("y.shape",y.shape)
print("x:", x, "onehot y:", y)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,shuffle=True, random_state=333, test_size = 0.2
)
# import sklearn
# print(sklearn.__version__) 
'''
onehot=encoder.fit_transform(df[['']])
#어떤 컬럼을 바꿀까 ? 
train_cat = ohe.fit_transform(x[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19','Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33','Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']])
print('y의 라벨 값 :',np.unique(y))

'''
#2. 모델 구성 

model=Sequential()
model.add(Dense(1000,activation="relu",input_dim=54))
model.add(Dense(400,activation="relu"))
model.add(Dense(8,activation="softmax"))
model.add(Dense(6,activation="linear"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="softmax"))
model.summary()
#서머리 양 갯수 
# Total params: 122,205
# Trainable params: 122,205
# Non-trainable params: 0

#3. 컴파일, 훈련 

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc',patience=50,mode='max',verbose=1,
                   restore_best_weights=True)
import time 
start_time = time.time()
model.fit(x_train,y_train,epochs=10,batch_size=100000,verbose=1,validation_split=0.2,callbacks=[es])
end_time =time.time()
#4. 평가, 예측
result=model.evaluate(x_test,y_test)
print('result : ',result )
print('loss : ',result[0] )
print('acc : ',result[1] )
print('걸린 시간은  : ',round(end_time - start_time,2))
"""
y_predict=model.predict(x_test)
y_test_acc = np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=-1)
print('y_test_acc : ',y_test_acc)
print('y_predict : ',y_predict )

acc = accuracy_score(y_test_acc,y_predict)


import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['val_acc'],marker='.',c='red',label='val_loss') # 뭔가 명시하지 않아도 된다는데 
plt.plot(hist.history['acc'],marker='.',c='blue',label='loss') # 뭔가 명시하지 않아도 된다는데 
plt.title('asd') #이름 지어주기
plt.xlabel('epochs')
plt.ylabel('loss,val_loss')
plt.legend()
plt.grid()
plt.show()
"""

# 결정 계수 - 보조 지표 - R2
# ACCURACY - 정확도 
# 대회에서 평가 지표를 잘 봐야한다. 
# 데이터 셋을 나눌 때는 트레인 , 테스트 
# 트레인에서 트레인 발류데이션 
# 인공지능이란 학습시킨후 예측 하게 만드는 것 
# 인공지능 잘 만들기 위해서는 좋은 데이터를 많이 가지고 있어야한다.
# 얼리 스탑핑
# 컷 시키는 지점이 가장 작지 않을 수도 있다. 
