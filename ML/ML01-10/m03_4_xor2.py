'''
# 인공지능 겨울
# 전기 회로에서 하는 and or xor
# and : 두 개 곱하기
and 0    1
0   0    0
1   0    1
이런식 거짓하나만 잇어도 거짓이다.
1이 두 개가 있어야 1이다. 
and는 선그으면 확실 해서 찾을 수 있다. 
# or : 두 개 더하기
or  0    1
0   0    1
1   1    1
1이 하라도 있으면 1이다. 
S는 선그으면 확실 해서 찾을 수 있다. 
# xor : 둘이 다르면 1 같으면 0 
# 반전 - 이거 해결 못해서 인공지능 겨울이 옮
xor 0    1
0   0    1
1   1    0
어떻게 선을 그어도 최대가 75%다.
평면이 아니라 입체 형태로 생각해서 접으면 된다.
하지만 그 때 시절에 좋은 장비가 어려웠다. 
# 인공지능 첫 번째 겨울 : https://noros.tistory.com/110
# 그 때 당시 vms이였음
# 선하나 그어서 해결했는데 아니여서 선 못그어서 망할 뻔 했다.
'''
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

# 2. 모델
# model = LinearSVC()
model = SVC()
# 

# 3. 훈련
model.fit(x_data,y_data)

# 4. 평가 예측
y_pred = model.predict(x_data)
results = model.score(x_data,y_data)
# 분류인지 회귀인지 모델에서 알아서 적용해줌 
print("model.score : ",results)
acc = accuracy_score(y_data,y_pred)
print('accuracy_score : ',acc)
