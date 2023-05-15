import math
import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,accuracy_score
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')

path = "./_data/dacon_crime/"
path_save = "./_save/dacon_crime/"


#1. 데이터

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(871393, 9)

test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) # y가 없다. train_csv['Calories_Burned']
print(test_csv.shape) # (159621, 8)

print(train_csv.info()) 
'''
#   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   월        84406 non-null  int64
 1   요일       84406 non-null  object
 2   시간       84406 non-null  int64
 3   소관경찰서    84406 non-null  int64
 4   소관지역     84406 non-null  float64
 5   사건발생거리   84406 non-null  float64
 6   강수량(mm)  84406 non-null  float64
 9   풍향       84406 non-null  float64
 10  안개       84406 non-null  float64
 11  짙은안개     84406 non-null  float64
 12  번개       84406 non-null  float64
 13  진눈깨비     84406 non-null  float64
 14  서리       84406 non-null  float64
 15  연기/연무    84406 non-null  float64
 16  눈날림      84406 non-null  float64
 17  범죄발생지    84406 non-null  object
 18  TARGET   84406 non-null  int64
dtypes: float64(13), int64(4), object(2)

'''
print(train_csv.columns)
'''
Index(['월', '요일', '시간', '소관경찰서', '소관지역', '사건발생거리', '강수량(mm)', '강설량(mm)',
       '적설량(cm)', '풍향', '안개', '짙은안개', '번개', '진눈깨비', '서리', '연기/연무', '눈날림',
       '범죄발생지', 'TARGET'],
      dtype='object')
''' 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의
le_list = ['요일', '범죄발생지']
csv_all = pd.concat([train_csv,test_csv],axis=0)
print(csv_all.columns)
for i,v in enumerate(le_list) :
    csv_all[v] = le.fit_transform(csv_all[v]) # 0과 1로 변화
    train_csv[v] = le.transform(train_csv[v]) # 0과 1로 변화
    test_csv[v] = le.transform(test_csv[v]) # 0과 1로 변화
    print(f'{i}번째',np.unique(csv_all[v]))
    


x = train_csv.drop(['TARGET'],axis=1)#(1328, 9)
print("x : ", x) # [7500 rows x 7 columns]

# y = train_csv['Book-Rating']
y = to_categorical(train_csv['TARGET'])
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,random_state=8715
)



# 2. 모델 구성
start = time.time()
model = Sequential()
model.add(Dense(64,activation='relu',input_shape=(x_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

scaler = StandardScaler()
# scaler = MaxAbsSc aler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("x_train.shape : ",x_train.shape)
print("y_train.shape : ",y_train.shape)

# 3. 훈련
model.compile(optimizer='adam', loss='categorical_crossentropy')
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=25,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,
          epochs=256,batch_size=8000,
          validation_split=0.2,
          callbacks=[es]
          )
end = time.time()
print('걸린시간:',round(end-start,2))
model.save(path_save+'04')
# 4. 평가, 예측 
# print("최상의 매개변수 : ",model.best_params_)
# print("최상의 매개변수 : ",model.best_score_)
result = model.evaluate(x_test,y_test)
print(" result: ", result)
y_pred= model.predict(x_test)
print('y_test : ', y_test.shape) 
print('y_pred : ', y_pred.shape)
y_pred=np.argmax(y_pred,axis=1)
y_test=np.argmax(y_test,axis=1)
print('y_test : ', y_test.shape) 
print('y_pred : ', y_pred.shape) 
print('y_test : ', y_test) 
print('y_pred : ', y_pred) 

acc = accuracy_score(y_test,y_pred)
print('acc :',acc)


# 5. 제출

submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
print(submission.shape)
y_submit = model.predict(test_csv)
y_submit=np.argmax(y_submit,axis=1)
print(y_submit.shape)
submission['TARGET'] =y_submit
submission.to_csv(path_save+'submit0515_02.csv')
