# 수정요망
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential #Sequential모델 
from tensorflow.keras.layers import Dense #Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error #mse에서 루트 씌우면 rmse로 할 수 있을지도?
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
import time
def rmse(y_test,y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M")  

#1. 데이터

path = "./_data/dacon_kcal/"
path_save = "./_save/dacon_kcal/"

train_csv=pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(7500, 10)

test_csv=pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv) # y가 없다. train_csv['Calories_Burned']
print(test_csv.shape) # (7500, 9)
print(train_csv.columns)
'''
Index(['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)',
       'Height(Remainder_Inches)', 'Weight(lb)', 'Weight_Status', 'Gender',
       'Age', 'Calories_Burned'],dtype='object')
'''

####################################### 라벨 인코딩 #####################################

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 정의

# Gender
le.fit(train_csv['Gender']) #
aaa = le.transform(train_csv['Gender']) # 0과 1로 변화
print(aaa.shape)
print(type(aaa))
#print(np.unique(aaa,return_count=True))
train_csv['Gender'] = aaa #다시 집어넣기
print(train_csv)
test_csv['Gender'] = le.transform(test_csv['Gender'])
print(np.unique(aaa))

#Weight_Status
le.fit(train_csv['Weight_Status']) #
bbb = le.transform(train_csv['Weight_Status']) # 0과 1로 변화
print(bbb.shape)
print(type(bbb))
# print(np.unique(aaa,return_count=True))
train_csv['Weight_Status'] = bbb #다시 집어넣기
# print(train_csv)
test_csv['Weight_Status'] = le.transform(test_csv['Weight_Status'])
print(np.unique(bbb)) #[0 1 2]


#####################train_csv 데이터에서 x와 y를 분리#######################

x = train_csv.drop(['Calories_Burned'],axis=1)#(1328, 9)
print("x : ", x) # [7500 rows x 7 columns]

y = train_csv['Calories_Burned']
print(y) # Name: Calories_Burned, Length: 7500, dtype : float64
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,random_state=8715
)

print("x_train.shape : ",x_train.shape) #(6750, 9)
print("y_train.shape : ",y_train.shape) #(6750,)
print("x_test.shape : ",x_test.shape) #(750, 9)
print("y_test.shape : ",y_test.shape) # (750,)

paramiters = [
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10]},
    {"n_estimators" : [100,200],"max_depth" : [6,8,10,12],"min_samples_leaf" : [3,5,7,10],"min_samples_split" : [2,3,5,10]},
    {"n_estimators" : [100,200],"min_samples_leaf" : [3,5,7,10]},
    {"n_estimators" : [100,200]},
]

# 2. 모델 
model = GridSearchCV(ExtraTreesRegressor(),
                     paramiters, # 52번돌림
                     cv=5,
                     verbose=1,
                     n_jobs=-1) # 총260번 돌림
                     
# 3. 컴파일, 훈련
start_time = time.time()

model.fit(x_train,y_train)
print("최적의 매개변수 : ",model.best_estimator_) 
print("최적의 파라미터 : ",model.best_params_)
print("best_score_ : ",model.best_score_) # train의 베스트 스코어 
print("model.score : ",model.score(x_test,y_test)) # test의 베스트 스코어 

end_time = time.time()
print('걸린 시간 : ',np.round(end_time-start_time))

y_pred = model.predict(x_test)
print('r2_score : ',r2_score(y_test,y_pred))
y_pred_best= model.best_estimator_.predict(x_test)
print('최적의 튠  r2_score : ',r2_score(y_test,y_pred_best))
