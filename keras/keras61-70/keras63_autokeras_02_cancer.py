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
'''
gpt
acc = accuracy_score(y_test,y_predict)가 0.0 나오는 이유 
두 번째 인자로 전달된 y_predict는 Autokeras의 모델에서 반환된 예측값입니다.
그러나 y_predict는 모델의 예측 결과가 아닌, model.predict()의 반환값 자체를 사용하고 있습니다.
Autokeras에서 model.predict()의 반환값은 2차원 배열로 나오는데, 
이는 각 클래스에 대한 확률을 나타내기 위한 softmax 출력입니다. 
accuracy_score 함수는 클래스 레이블을 비교하기 위해
1차원 배열 형태의 예측값을 기대합니다.
따라서 정확도를 계산하기 위해서는 
y_predict를 1차원 배열로 변환한 후에
accuracy_score를 사용해야 합니다. 
아래와 같이 코드를 수정하면 됩니다
'''
print('걸린시간 : ',round(end-start,2))

print(model.__class__.__name__,'의 acc : ',results[1] )
# StructuredDataClassifier 의 acc :  [0.757175087928772, 0.800000011920929]
'''
loss :  0.7777271270751953
acc :  0.8333333134651184
acc2 :  0.0
걸린시간 :  5.96
StructuredDataClassifier 의 acc :  0.8333333134651184
'''
