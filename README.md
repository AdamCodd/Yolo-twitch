# Yolo-twitch
A Python application that performs real-time object detection, segmentation, classification, and pose estimation on Twitch streams using YOLO models. The application supports low-latency streaming with audio playback capabilities.

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
- CUDA-compatible GPU (recommended)
- See requirements.txt for Python dependencies

## Quick Start
```python
streamer = "username"  # Twitch username
hls_url = get_low_latency_stream(streamer)

detector = VideoDetector(
    task="detect",
    size="nano",
    input_path=hls_url,
    class_name="person",
    enable_audio=True
)
detector.process_video()
