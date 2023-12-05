import cv2

input_img = './inputs/'
output_img = './outputs/'

# 이미지 불러오기
image = cv2.imread(input_img)

# 이미지 크기 변경 (이미지, 크기, interpolation)
upscaled_image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

# 확대된 이미지 저장
cv2.imwrite(output_img, upscaled_image)
