# 아마 틀릴듯 수정 요망
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

# scaler = StandardScaler()
# # scaler = MinMaxScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print("x_train.shape01: ",x_train.shape)
# print("y_train.shape01 : ",y_train.shape)

####################################### 상관관계 찾기 #####################################
# import matplotlib.pyplot as plt
# import seaborn as sns

# print(test_csv.corr())Epoch 00074: early stopping
# plt.figure(figsize=(10,8))
# sns.set(font_scale=1.2)
# sns.heatmap(train_csv.corr(),square=True, annot=True,cbar=True)
# plt.show()
# 가장 낮은 거 
# Height(Remaineder_Inches)
# Age

# 2. 모델 구성

allAlgoritms = all_estimators(type_filter='regressor')

# print('allAlgoritms : ',allAlgoritms)
# 튜플 안에 첫번째는 스트링 형태의 모델, 두번째는 클래스로 정의된 모델
# print(len(allAlgoritms)) #55

max_r2 = 0
max_name = '바보'
for (name, algoritm) in allAlgoritms :
    try: # 에외처리 
        model = algoritm()
        # 3. 훈련
        model.fit(x_train,y_train) 
        # 4. 평가, 예측
        results = model.score(x_test,y_test)
        print( name ,'의 정답률 : ',results)
        if max_r2 < results:
            max_r2 = results
            max_name = name
            y_predict=model.predict(x_test)

    except :  
        print(name,' : 에러 모델')

print('===================================')
print('최고 모델 : ',max_name, max_r2)
print('===================================')
'''

y_submit = model.predict(test_csv)=
rmse = rmse(y_test,y_predict) #함수 실행

#Calories_Burned 넣기 
submission  = pd.read_csv(path+'sample_submission.csv',index_col=0)
#print(submission)
submission['Calories_Burned'] =y_submit
submission.to_csv(path_save+date+'plus'+'submit04.csv')

'''

