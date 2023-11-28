import torch

# 시스템에서 사용 가능한 GPU 디바이스 수를 얻기
gpu_count = torch.cuda.device_count()
print(f"시스템에서 사용 가능한 GPU 디바이스 수: {gpu_count}")
