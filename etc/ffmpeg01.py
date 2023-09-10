import subprocess

def run_ffmpeg_command(command):
    """Execute the given ffmpeg command."""
    subprocess.run(command, check=True)

def preprocess_video(input_path, output_path):
    """Preprocess video by changing resolution, audio sampling rate, and extracting a specific time range."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', 'scale=1280:720',  # Change resolution to 1280x720
        '-ar', '44100',  # Change audio sampling rate to 44.1 kHz
        '-ss', '00:01:00',  # Start from 1 minute
        '-t', '00:02:00',  # Extract 2 minutes duration
        output_path
    ]
    run_ffmpeg_command(command)

def postprocess_video(input_path, watermark_path, output_path):
    """Postprocess video by adding a watermark and adjusting audio volume."""
    command = [
        'ffmpeg',
        '-i', input_path,
        '-i', watermark_path,
        '-filter_complex', 'overlay=W-w-10:H-h-10',  # Place watermark at the bottom right corner
        '-af', 'volume=1.5',  # Increase audio volume by 1.5 times
        output_path
    ]
    run_ffmpeg_command(command)

# Example usage:
# preprocess_video('input.mp4', 'preprocessed.mp4')
# postprocess_video('preprocessed.mp4', 'watermark.png', 'final_output.mp4')
