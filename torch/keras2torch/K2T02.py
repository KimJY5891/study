
# 1. 데이터
import numpy as np
import torch
x = np.array([1,2,3])
y = np.array([1,2,3])

# numpy배열을 pytorch 텐서로 변환 
x_tensor = torch.tensor(x,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)
print(x_tensor.size(),y_tensor.size()) # torch.Size([3]) torch.Size([3])

# 2. 모델 구성
import torch.nn as nn
import torch.optim as optim
class LinearRegression(nn.Module) :
    def __init__(self) : 
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1)  # 입력 차원: 1, 출력 차원: 1

    def forward(self,x) :
        return self.linear(x)

model = LinearRegression()

# 3. 컴파일 훈련 
criterion = nn.MSELoss() # 손실 함수 : 평균 제곱 오차
optimizer =optim.Adam(model.parameters(),lr=0.01)

# 훈련 루프 
num_epochs = 256 
for epoch in range(num_epochs) :
    # 순전파
    outputs = model(x_tensor.view(-1,1)) #모델에 입력을 전달

    # 손실 계산 
    loss = criterion(outputs,y_tensor.view(-1,1))

    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    # optimizer.zero_grad() 함수는 PyTorch에서 모델의 경사(gradient)를 초기화하는 역할을 합니다.
    # 이 함수는 주로 각 미니배치(또는 에포크)의 시작 부분에서 호출되어야 합니다.
    loss.backward() # 역전파: 경사 계산
    optimizer.step() # 파라미터 업데이트
# 4. 예측 
x_tensor.view(-1,1)
predicted = model(x_tensor.view(-1,1))
print(f"inputs : {x_tensor.view(-1,1)}")
print(f"모델 예측 결과 : {predicted}")
