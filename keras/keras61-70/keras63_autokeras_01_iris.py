import autokeras as ak
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score,r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
import pandas as pd 


# 1. 데이터 

path = "c:/study/_data/dacon_diabets/"
path_save = "c:/study/_save/dacon_diabets/"
train_csv=pd.read_csv(path + 'train.csv',index_col=0)
test_csv=pd.read_csv(path + 'test.csv',index_col=0)
train_csv = train_csv.dropna()

x = train_csv.drop(['Outcome'],axis=1)
y = train_csv['Outcome']
print("x.shape : ",x.shape)#(652, 8)
print("y.shape : ",y.shape)#(652, )

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size= 0.8, random_state = 337, shuffle=True, stratify=y
)
print('x_test : ',x_test.shape)
print('x_train : ',x_train.shape)

# 2. 모델 
model = ak.StructuredDataClassifier(
    # column_names=[], # 선택 사항
    # column_types={}, # 선택 사항
    max_trials=1, # 10개의 다른 모델 
    overwrite=False
)

# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs = 10, validation_split = 0.15)
end = time.time()

# 4. 평가 예측

y_pred = model.predict(x_test)
y_pred = y_pred.reshape(131,)
results = model.evaluate(x_test, y_test)
acc = accuracy_score(y_test,y_pred)
print('loss : ',results[0])
print('acc : ',results[1])
print('걸린시간 : ',round(end-start,2))
print(model.__class__.__name__,'의 acc : ',acc )
# StructuredDataClassifier 의 acc :  [0.757175087928772, 0.800000011920929]
