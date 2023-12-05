import cv2

# 동영상 파일 열기
video_capture = cv2.VideoCapture('input_video.mp4')

# 동영상의 프레임 속성 가져오기
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

# 업스케일할 배율 설정
scale_factor = 2.0  # 확대 배율

# 출력 동영상 설정
output_width = int(frame_width * scale_factor)
output_height = int(frame_height * scale_factor)
output_video = cv2.VideoWriter('upscaled_video.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (output_width, output_height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # 프레임 업스케일
    upscaled_frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    
    # 업스케일된 프레임을 출력 동영상에 쓰기
    output_video.write(upscaled_frame)

    cv2.imshow('Upscaled Video', upscaled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 해제
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
