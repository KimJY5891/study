import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# 1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]] # (4,2)
y_data = [0,1,1,0] # (4,)

# 2. 모델
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(40,input_dim=2,activation='relu')) #퍼셉트론 케라스로 구현 
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1,activation='sigmoid'))
'''
model.score :  1.0
accuracy_score :  1.0
'''
# y는 (4,1)
# 히든 없이

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=30,mode='min',
               verbose=1,restore_best_weights=True)
model.fit(x_data,y_data,batch_size=1,epochs=100,callbacks=[es])

# 4. 평가 예측
y_pred = model.predict(x_data)
results = model.evaluate(x_data,y_data)
print("model.score : ",results[1])

acc = accuracy_score(y_data,np.round(y_pred)) # 둘 다 np로 맞춰줘야함 
print('accuracy_score : ',acc)
