import autokeras as ak 
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score,r2_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection  import train_test_split

# 1. 데이터 
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size= 0.8, random_state = 337, shuffle=True, stratify=y
)
print('x_test : ',x_test.shape)
print('x_train : ',x_train.shape)

# 2. 모델 
model = ak.StructuredDataClassifier(
    # column_names=[], # 선택 사항
    # column_types={}, # 선택 사항
    max_trials=10, # 10개의 다른 모델 
    overwrite=False
)

# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, epochs = 10, validation_split = 0.15)
end = time.time()

# 4. 평가 예측

y_predict = model.predict(x_test)
print(y_predict.shape)
y_predict = y_predict.reshape(y_predict.shape[0],)
print('y_predict : ',y_predict)
results = model.evaluate(x_test, y_test)
print('loss : ',results[0])
print('acc : ',results[1])
acc = accuracy_score(y_test,y_predict)
print('acc2 : ',acc)    
