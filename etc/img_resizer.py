from PIL import Image
import os

# 이미지 폴더 경로
image_folder = 'D:/20to60_asian_train_datasets/val60_data/'

# 변경된 이미지 저장 폴더 경로
output_folder = 'D:/20to60_asian_train_datasets/val60/'

# 새로운 이미지 크기
new_width = 256
new_height = 256

# 이미지 폴더 내의 이미지 파일 목록 가져오기
image_files = os.listdir(image_folder)

# 각 이미지에 대해 크기 변경 후 저장
for file in image_files:
    # 이미지 파일 경로
    image_path = os.path.join(image_folder, file)
    
    # 이미지 로드
    image = Image.open(image_path)
    
    # 이미지 크기 변경
    resized_image = image.resize((new_width, new_height))
    
    # 저장할 이미지 파일 경로
    output_path = os.path.join(output_folder, file)
    
    # 변경된 이미지 저장
    resized_image.save(output_path)
print('완료')
