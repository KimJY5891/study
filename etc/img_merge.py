import os
import cv2
import numpy as np

# train20 폴더와 train60 폴더 경로 설정
train20_folder = 'D:/20to60_asian_train_datasets/train20'
train60_folder = 'D:/20to60_asian_train_datasets/train60'

# train 폴더 경로 설정
train_save_folder = 'D:/20to60_asian_train_datasets/train_save'

# train20 폴더의 이미지 파일 목록 가져오기
train20_files = sorted(os.listdir(train20_folder))

# train60 폴더의 이미지 파일 목록 가져오기
train60_files = sorted(os.listdir(train60_folder))

# 이미지를 순차적으로 붙일 빈 캔버스 생성
canvas = np.zeros((256, 256*2, 3), dtype=np.uint8)

# train20 이미지와 train60 이미지를 순차적으로 붙이고 저장하기
for i in range(len(train20_files)):
    # train20 이미지 읽어오기
    train20_image = cv2.imread(os.path.join(train20_folder, train20_files[i]))
    # train60 이미지 읽어오기
    train60_image = cv2.imread(os.path.join(train60_folder, train60_files[i]))

    # 이미지를 캔버스에 붙이기
    canvas[:, :256, :] = train20_image
    canvas[:, 256:, :] = train60_image

    # 이미지 저장하기
    save_path = os.path.join(train_save_folder, f'combined_{i}.jpg')
    cv2.imwrite(save_path, canvas)

    # 이미지 보여주기
    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)

cv2.destroyAllWindows()
'''
# train20 폴더와 train60 폴더 경로 설정
train20_folder = 'D:/20to60_asian_train_datasets/train20_data'
train60_folder = 'D:/20to60_asian_train_datasets/train60_data'


# train20 폴더의 이미지 파일 목록 가져오기
train20_files = sorted(os.listdir(train20_folder))

# train60 폴더의 이미지 파일 목록 가져오기
train60_files = sorted(os.listdir(train60_folder))

# 이미지를 순차적으로 붙일 빈 캔버스 생성
canvas = np.zeros((256, 256*2, 3), dtype=np.uint8)

# train20 이미지와 train60 이미지를 순차적으로 붙이기
for i in range(len(train20_files)):
    # train20 이미지 읽어오기
    train20_image = cv2.imread(os.path.join(train20_folder, train20_files[i]))
    # train60 이미지 읽어오기
    train60_image = cv2.imread(os.path.join(train60_folder, train60_files[i]))

    # 이미지를 캔버스에 붙이기
    canvas[:, :256, :] = train20_image
    canvas[:, 256:, :] = train60_image

    # 이미지 보여주기
    cv2.imshow('Canvas', canvas)
    cv2.waitKey(0)

cv2.destroyAllWindows()
'''
