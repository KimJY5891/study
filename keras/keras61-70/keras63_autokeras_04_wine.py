import autokeras as ak 
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score,r2_score
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection  import train_test_split

# 1. 데이터 
x,y = load_wine(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(
    x,y, train_size= 0.8, random_state = 337, shuffle=True, stratify=y
)
print('x_test : ',x_test.shape)
print('x_train : ',x_train.shape)

# 2. 모델 
model = ak.StructuredDataClassifier(
    # column_names=[], # 선택 사항
    # column_types={}, # 선택 사항
    max_trials=100, # 10개의 다른 모델 
    overwrite=False
)

# 3. 컴파일, 훈련
start = time.time()
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau # 러닝레이트를 감축시켜라 
# es = EarlyStopping(monitor = 'val_loss',patience = 20, mode = 'min', verbose = 1)
model.fit(x_train, y_train, epochs = 1000, validation_split = 0.15,
          # callbacks=[es]
          )
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

print('걸린시간 : ',round(end-start,2))

print(model.__class__.__name__,'의 acc : ',results[1] )
# StructuredDataClassifier 의 acc :  [0.757175087928772, 0.800000011920929]
'''

'''
