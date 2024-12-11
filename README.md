# Yolo-twitch
A Python application that performs real-time object detection, segmentation, classification, and pose estimation on Twitch streams using YOLO models. The application supports low-latency streaming with audio playback capabilities. 

On practical use, my laptop with an Nvidia GeForce GTX 1650 and Intel i5 can manage a real-time stream at 360p/30FPS with sound with the nano model.

![Twitch detection](https://i.imgur.com/Qwn4KvC.png)

## Features
- Real-time video processing using YOLO v11 models
- Multiple detection tasks: object detection, segmentation, classification, and pose estimation
- Low-latency Twitch stream capture
- Optional audio playback
- Multi-threaded processing for improved performance
- CUDA acceleration support
- Configurable model sizes (nano, small, medium, large, xlarge)

## Requirements
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) for video processing. Ensure FFmpeg is installed and accessible in your system's PATH.
- CUDA-compatible GPU (recommended)
- See requirements.txt for Python dependencies

## Quick Start
```python
streamer = "username"  # Twitch username
hls_url = get_low_latency_stream(streamer)

detector = VideoDetector(
    task="detect", # segment, classify, pose
    size="nano",
    input_path=hls_url,
    class_name="person",
    enable_audio=True
)
detector.process_video()
