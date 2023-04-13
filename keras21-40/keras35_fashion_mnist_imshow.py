#import fashion_mnist -
# mnist는
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing  import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


print((x_train.shape,y_train.shape),(x_test.shape,y_test.shape)) #((60000, 28, 28), (60000,)) ((10000, 28, 28), (10000,))




import matplotlib.pyplot as plt
plt.imshow(x_train[0],'rgb') # 그림 보여줌
plt.show()


keras36_kaggle_house.py
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler #전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler #전처리

#1. 데이터

path = "./_data/kaggle_house/"
path_save = "./_save/kaggle_house/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)

####################################### LabelEncoder #######################################                                                                                 #
print("np.unique(train_csv['MSZoning']) : ",np.unique(train_csv['MSZoning'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
le.fit(train_csv['MSZoning']) 
aaa = le.transform(train_csv['MSZoning']) 
print(aaa)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa))
train_csv['MSZoning'] = aaa #다시 집어넣기
print(train_csv)
test_csv['MSZoning'] = le.transform(test_csv['MSZoning'])
#
print("np.unique(train_csv['Street']) : ",np.unique(train_csv['Street'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le02 = LabelEncoder() 
le02.fit(train_csv['Street']) 
aaa02 = le02.transform(train_csv['Street']) 
print(aaa02)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa02))
train_csv['Street'] = aaa02 #다시 집어넣기
print(train_csv)
test_csv['Street'] = le02.transform(test_csv['Street'])
#03
print("np.unique(train_csv['Alley']) : ",np.unique(train_csv['Alley'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le03 = LabelEncoder() 
le03.fit(train_csv['Alley']) 
aaa03 = le03.transform(train_csv['Alley']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['Alley'] = aaa03 #다시 집어넣기
print(train_csv)
test_csv['Alley'] = le03.transform(test_csv['Alley'])
#04
print("np.unique(train_csv['LotShape']) : ",np.unique(train_csv['LotShape'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le04 = LabelEncoder() 
le04.fit(train_csv['LotShape']) 
aaa04 = le04.transform(train_csv['LotShape']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa04))
train_csv['LotShape'] = aaa04 #다시 집어넣기
print(train_csv)
test_csv['LotShape'] = le04.transform(test_csv['LotShape'])
#05
print("np.unique(train_csv['LandContour']) : ",np.unique(train_csv['LandContour'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le05 = LabelEncoder() 
le05.fit(train_csv['LandContour']) 
aaa05 = le05.transform(train_csv['LandContour']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa05))
train_csv['LandContour'] = aaa05 #다시 집어넣기
print(train_csv)
test_csv['LotShape'] = le05.transform(test_csv['LotShape'])
#06
print("np.unique(train_csv['Utilities']) : ",np.unique(train_csv['Utilities'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le06 = LabelEncoder() 
le06.fit(train_csv['Utilities']) 
aaa06 = le06.transform(train_csv['Utilities']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa06) : ",np.unique(aaa06))
train_csv['Utilities'] = aaa06 #다시 집어넣기
print(train_csv)
test_csv['Utilities'] = le06.transform(test_csv['Utilities'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#08
print("np.unique(train_csv['LandSlope']) : ",np.unique(train_csv['LandSlope'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le08 = LabelEncoder() 
le08.fit(train_csv['LandSlope']) 
aaa08 = le08.transform(train_csv['LandSlope']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa08))
train_csv['LandSlope'] = aaa08 #다시 집어넣기
print(train_csv)
test_csv['LandSlope'] = le08.transform(test_csv['LandSlope'])

#09
print("np.unique(train_csv['Neighborhood']) : ",np.unique(train_csv['Neighborhood'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le09 = LabelEncoder() 
le09.fit(train_csv['Neighborhood']) 
aaa09 = le09.transform(train_csv['Neighborhood']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['Neighborhood'] = aaa09 #다시 집어넣기
print(train_csv)
test_csv['Neighborhood'] = le09.transform(test_csv['Neighborhood'])

#10
print("np.unique(train_csv['Condition1']) : ",np.unique(train_csv['Condition1'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le10 = LabelEncoder() 
le10.fit(train_csv['Condition1']) 
aaa10 = le10.transform(train_csv['Condition1']) 
print(aaa10)#[0 1 2 3 4]
print("np.unique(aaa10) : ",np.unique(aaa03))
train_csv['Condition1'] = aaa10 #다시 집어넣기
print(train_csv)
test_csv['Condition1'] = le10.transform(test_csv['Condition1'])

#11
print("np.unique(train_csv['Condition2']) : ",np.unique(train_csv['Condition1'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le11 = LabelEncoder() 
le11.fit(train_csv['Condition2']) 
aaa11 = le11.transform(train_csv['Condition2']) 
print(aaa11)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa11))
train_csv['Condition2'] = aaa11 #다시 집어넣기
print(train_csv)
test_csv['Condition2'] = le11.transform(test_csv['Condition2'])

#12
print("np.unique(train_csv['BldgType']) : ",np.unique(train_csv['BldgType'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le12 = LabelEncoder() 
le12.fit(train_csv['BldgType']) 
aaa12 = le12.transform(train_csv['BldgType']) 
print(aaa12)#[0 1 2 3 4]
print("np.unique(aaa12) : ",np.unique(aaa12))
train_csv['BldgType'] = aaa12 #다시 집어넣기
print(train_csv)
test_csv['BldgType'] = le12.transform(test_csv['BldgType'])

#13
print("np.unique(train_csv['HouseStyle']) : ",np.unique(train_csv['HouseStyle'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le13 = LabelEncoder() 
le13.fit(train_csv['HouseStyle']) 
aaa13 = le13.transform(train_csv['HouseStyle']) 
print(aaa13)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['HouseStyle'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['HouseStyle'] = le07.transform(test_csv['HouseStyle'])

#14
print("np.unique(train_csv['RoofStyle']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le14 = LabelEncoder() 
le14.fit(train_csv['RoofStyle']) 
aaa14 = le14.transform(train_csv['RoofStyle']) 
print(aaa14)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa14))
train_csv['RoofStyle'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['RoofStyle'] = le14.transform(test_csv['RoofStyle'])

#15
print("np.unique(train_csv['RoofMatl']) : ",np.unique(train_csv['RoofMatl'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le15 = LabelEncoder() 
le15.fit(train_csv['RoofMatl']) 
aaa15 = le15.transform(train_csv['RoofMatl']) 
print(aaa15)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa15))
train_csv['RoofMatl'] = aaa15 #다시 집어넣기
print(train_csv)
test_csv['RoofMatl'] = le15.transform(test_csv['RoofMatl'])

#16
print("np.unique(train_csv['Exterior1st']) : ",np.unique(train_csv['Exterior1st'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le16 = LabelEncoder() 
le16.fit(train_csv['Exterior1st']) 
aaa16 = le16.transform(train_csv['Exterior1st']) 
print(aaa16)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa16))
train_csv['Exterior1st'] = aaa16 #다시 집어넣기
print(train_csv)
test_csv['Exterior1st'] = le16.transform(test_csv['Exterior1st'])


#17
print("np.unique(train_csv['Exterior2nd']) : ",np.unique(train_csv['Exterior2nd'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le17 = LabelEncoder() 
le17.fit(train_csv['Exterior2nd']) 
aaa17 = le17.transform(train_csv['Exterior2nd']) 
print(aaa17)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa17))
train_csv['Exterior2nd'] = aaa17 #다시 집어넣기
print(train_csv)
test_csv['Exterior2nd'] = le17.transform(test_csv['Exterior2nd'])

#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])
#07
print("np.unique(train_csv['LotConfig']) : ",np.unique(train_csv['LotConfig'])) # ['C (all)' 'FV' 'RH' 'RL' 'RM']
from sklearn.preprocessing import LabelEncoder
le07 = LabelEncoder() 
le07.fit(train_csv['LotConfig']) 
aaa07 = le07.transform(train_csv['LotConfig']) 
print(aaa03)#[0 1 2 3 4]
print("np.unique(aaa) : ",np.unique(aaa03))
train_csv['LotConfig'] = aaa07 #다시 집어넣기
print(train_csv)
test_csv['LotConfig'] = le07.transform(test_csv['LotConfig'])


"""
####################################### 결측치 처리 #######################################                                                                                 #
print(train_csv.isnull().sum())
# train_csv = train_csv.dropna()
train_csv.fillna(0, inplace=True)
print(train_csv.isnull().sum())
print(train_csv.info())
# print(train_csv.shape)

####################################### 결측치 처리 #######################################                                                                                 #
x = train_csv.drop(['SalePrice'],axis=1)#
y = train_csv['SalePrice']
print("x.shape : ",x.shape)#(1460, 79)
print("y.shape : ",y.shape)#(1460,)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.9,shuffle=True,random_state=1234
)
scaler = StandardScaler()
# #scaler = MinMaxScaler()
# #scaler = MaxAbsScaler()
# # scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
test_csv = scaler.transform(test_csv)

print("x.shape : ",x.shape)#
print("y.shape : ",y.shape)

#2. 모델 구성
model =Sequential()
model.add(Dense(200,activation="relu",input_dim=79))
model.add(Dropout(0.5))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="relu"))
model.add(Dense(16,activation="linear"))
model.add(Dense(24,activation="linear"))
model.add(Dense(100,activation="relu"))
model.add(Dense(200,activation="relu"))
model.add(Dense(400,activation="relu"))
model.add(Dense(800,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(80,activation="relu"))
model.add(Dense(40,activation="relu"))
model.add(Dense(20,activation="linear"))
model.add(Dense(10,activation="linear"))
model.add(Dense(8,activation="linear"))
model.add(Dense(6,activation="linear"))
model.add(Dense(4,activation="linear"))
model.add(Dense(2))
model.add(Dense(1,activation='linear'))



#3. 컴파일, 훈련 
model.compile(loss='mse',optimizer='adam',
              metrics=['mse',]#'mean_squard_error','acc']# 값 확인용 훈련에 영향을 미치지 않는다.
              )
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='mse',patience=160,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=3000,batch_size=1000,verbose=1,validation_split=0.2,callbacks=[es])

# 4. 평가, 예측
loss =model.evaluate(x_test,y_test) 
print('loss : ',loss )
y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)
print('r2스코어 : ',r2)

y_submit = np.round(model.predict(test_csv))
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
submission['SalePrice'] =y_submit
submission.to_csv(path_save+'submit_0317_01.csv')
#median*()
"""
