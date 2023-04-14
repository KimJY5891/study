'''
# 인공지능 겨울
# 전기 회로에서 하는 and or xor
# and : 두 개 곱하기
    0    1
0   0    0
1   0    1
이런식 거짓하나만 잇어도 거짓이다.
1이 두 개가 있어야 1이다. 
# or : 두 개 더하기
    0    1
0   0    1
1   1    1
1이 하라도 있으면 1이다. 
# xor : 반전 - 이거 해결 못해서 인공지능 겨울이 옮
# 인공지능 첫 번째 겨울 : https://noros.tistory.com/110
# 그 때 당시 vms이였음 
# 선하나 그어서 해결했는데 아니여서 선 못그어서 망할 뻔 했다.
'''
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,1]

# 2. 모델
model = LinearSVC()

# 3. 훈련
model.fit(x_data,y_data)

# 4. 평가 예측
y_pred = model.predict(x_data)
results = model.score(x_data,y_data)
# 분류인지 회귀인지 모델에서 알아서 적용해줌 
print("model.score : ",results)
acc = accuracy_score(y_data,y_pred)
print('accuracy_score : ',acc)
