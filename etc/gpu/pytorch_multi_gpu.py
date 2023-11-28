import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os

# CUDA_VISIBLE_DEVICES 설정 (여러 개의 GPU를 사용하려면 설정)
os.environ['CUDA_VISIBLE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# 데이터 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

# 모델 정의
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 모델 및 손실 함수, 최적화기 초기화
latent_dim = 30000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 GPU로 이동하고 nn.DataParallel로 감싸기
autoencoder = Autoencoder(latent_dim).to(device)
if torch.cuda.device_count() > 1:
    autoencoder = nn.DataParallel(autoencoder)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 모델 훈련
startTime = time.time()
num_epochs = 32
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in trainloader:
        inputs, _ = data
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}")

endTime = time.time()
print(f"걸린 시간: {endTime - startTime:.2f} 초")

# 재구성 이미지 생성 및 시각화
n = 20
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 이미지 표시
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(inputs[i].cpu().numpy().squeeze(), cmap='gray')
    plt.title('original')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성 이미지 표시
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap='gray')
    plt.title('reconstructed')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
