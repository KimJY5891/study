import subprocess
# conda install -c conda-forge ffmpeg
# pip install ffmpeg
def run_ffmpeg_command(command):
    # 지정된 fmpeg 명령을 실행
    subprocess.run(command, check=True)

def preprocess_video(input_path, output_path):
    """해상도, 오디오 샘플링 속도를 변경하고 특정 시간 범위를 추출하여 비디오를 전처리함"""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', 'scale=1280:720',  # 해상도를 1280x720으로 변경
        '-ar', '44100',  # 오디오 샘플링 속도를 44.1kHz로 변경
        '-ss', '00:01:00',  # 1분부터 시작
        '-t', '00:02:00',  # 지속시간 2분 추출
        output_path
    ]
    run_ffmpeg_command(command)

def postprocess_video(input_path, watermark_path, output_path):
    """워터마크를 추가하고 오디오 볼륨을 조정하여 비디오를 후처리함"""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-i', watermark_path,
        '-filter_complex', 'overlay=W-w-10:H-h-10',  # 오른쪽 아래 모서리에 워터마크 배치
        '-af', 'volume=1.5',  # 오디오 볼륨 1.5배 증가
        output_path
    ]
    run_ffmpeg_command(command)

# Example usage:
# preprocess_video('input.mp4', 'preprocessed.mp4')
# postprocess_video('preprocessed.mp4', 'watermark.png', 'final_output.mp4')
