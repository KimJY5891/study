from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision import datasets

import torch.nn as nn


'''
사용자 정의 nn.Module 사용하기
사용자 지정 nn.Module 클래스를 사용하여 복잡한 네트워크 구조를 만들 수 있습니다.

nn.Sequential 사용하기
nn.Sequential은 순차적으로 연산을 수행하는 컨테이너입니다. 이를 사용하여 여러 레이어를 쌓을 수 있습니다

nn.Sequential과 사용자 정의 nn.Module 사이의 주요 차이점은 유연성과 복잡성입니다.

nn.Sequential:

장점:
간결성: 간단한 레이어 구조를 빠르고 쉽게 생성할 수 있습니다.
명확성: 순차적인 레이어를 연결하기 위한 간편한 방법입니다.
단점:
유연성 부족: 레이어 간의 복잡한 상호 작용이나 조건부 연산 (예: skip connections, 여러 입력 및 출력을 갖는 네트워크)을 쉽게 구현하기 어렵습니다.
디버깅: 에러 발생 시, 특정 레이어를 지정하는 것이 사용자 정의 nn.Module에 비해 덜 직관적일 수 있습니다.
사용자 정의 nn.Module:

장점:
유연성: 레이어 간의 복잡한 관계, 다양한 입출력 구조, 조건부 로직 등을 쉽게 통합할 수 있습니다.
구조화: 큰 프로젝트나 복잡한 모델에서는 사용자 정의 nn.Module을 통해 더 구조화된 코드를 작성할 수 있습니다.
재사용성: 특정 연산 블록이나 레이어 구조를 별도의 nn.Module 서브클래스로 분리하여 재사용하기 용이합니다.
단점:
코드 길이: 간단한 모델에 대해서는 nn.Sequential에 비해 코드가 길어질 수 있습니다.
복잡성: 초보자에게는 처음에 약간 복잡하게 느껴질 수 있습니다.
결론:

간단한, 순차적인 네트워크 구조의 경우 nn.Sequential을 사용하는 것이 좋습니다.
복잡한 네트워크 구조나 특정 로직이 필요한 경우, 사용자 정의 nn.Module을 사용하는 것이 좋습니다.

'''

# 모듈 
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 첫 번째 Conv 레이어
        # 입력: 3 채널 (RGB), 출력: 16 채널, 필터 크기: 3x3, stride: 1, padding: 1
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # Max pooling with 2x2 window
        
        # 두 번째 Conv 레이어
        # 입력: 16 채널, 출력: 32 채널, 필터 크기: 3x3, stride: 1, padding: 1
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        return x
# 시퀀셜

layers = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1),
    nn.ReLU(),
    nn.Conv2d(20, 50, 5, 1),
    nn.ReLU()
)
layers = layers.to(torch.device('cpu'))


