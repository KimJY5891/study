# 1. 데이터 
import numpy as np
import torch
x = np.array([1,2,3])
y = np.array([1,2,3])

# numpy 배열을 pytorch 텐서로 변환 
x_tensor = torch.tensor(x,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)

# 2. 모델 
import torch.nn as nn
class keras2torch(nn.Module) : 
    def __init__(self) : 
    # __init__ 메서드는 파이썬 클래스에서 특별한 메서드로, 객체가 생성될 때 자동으로 호출되는 초기화 메서드
    # 이 메서드를 사용하여 클래스의 속성(attribute)을 초기화하거나 다양한 설정 작업을 수행할 수 있습니다.
        super(keras2torch, self).__init__() 
        # 부모 클래스인 nn.Module의 초기화 메서드 호출
        # 여기서 모델의 레이어들을 초기화하고 설정합니다.
        
         # 나머지 레이어들도 유사하게 초기화됩니다.
