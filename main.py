from ultralytics import YOLO
import cv2
import os
import streamlink
import ffmpeg
import numpy as np
from threading import Thread
from queue import Queue, Empty
import time
import pyaudio

class VideoDetector:
    COCO_CLASSES = {
    'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
    'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9,
    'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
    'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19,
    'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24,
    'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29,
    'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
    'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38,
    'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43,
    'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48,
    'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53,
    'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58,
    'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63,
    'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68,
    'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73,
    'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78,
    'toothbrush': 79
    }

    MODEL_SIZES = {
        "nano": "n",
        "small": "s",
        "medium": "m",
        "large": "l",
        "xlarge": "x"
    }
    
    TASKS = {
        "detect": "",
        "segment": "-seg",
        "classify": "-cls",
        "pose": "-pose"
    }


    def __init__(self, input_path: str, size: str = "nano", task: str = "detect", 
                 class_name: str = None, enable_audio: bool = False):
        if size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid size. Must be one of: {list(self.MODEL_SIZES.keys())}")
        if task not in self.TASKS:
            raise ValueError(f"Invalid task. Must be one of: {list(self.TASKS.keys())}")
            
        size_code = self.MODEL_SIZES[size]
        task_suffix = self.TASKS[task]
        model_name = f"yolo11{size_code}{task_suffix}.pt"
        model_folder = os.path.dirname(os.path.abspath(__file__)) # Save the model where the script is located
        model_path = os.path.join(model_folder, model_name)
    
        self.model = YOLO(model_path, task=task)
        self.input_path = input_path
        self.class_id = None
        self.device = 'cuda:0'
        self.enable_audio = enable_audio
        
        if class_name:
            if class_name not in self.COCO_CLASSES:
                raise ValueError(f"Invalid class name. Must be one of: {list(self.COCO_CLASSES.keys())}")
            self.class_id = self.COCO_CLASSES[class_name]

        self.frame_buffer = Queue(maxsize=10) # 10 for 360p
        self.processed_frames = Queue(maxsize=5) # 5 for 360p
        self.audio_buffer = Queue(maxsize=5) # 5 for 360p
        self.running = False
        self.last_frame_time = 0
        self.target_fps = 30
        
        # Audio setup
        self.audio = pyaudio.PyAudio() if enable_audio else None
        self.audio_stream = None
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 1024

    def setup_audio_stream(self):
        if not self.enable_audio:
            return
        self.audio_stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

    def read_stream(self):
        try:
            # Get stream info to determine resolution
            probe = ffmpeg.probe(self.input_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            frame_size = width * height * 3

            # Create video process
            video_process = (
                ffmpeg
                .input(
                    self.input_path,
                    flags='low_delay',
                    fflags='nobuffer',
                    thread_queue_size=1024,
                    hwaccel='cuda'
                )
                .output(
                    'pipe:',
                    format='rawvideo',
                    pix_fmt='rgb24',
                    acodec='none',
                    vsync='drop',
                    max_delay=100000,
                    preset='ultrafast',
                    tune='zerolatency',
                    loglevel='error'
                )
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            # Create audio process only if audio is enabled
            audio_process = None
            audio_thread = None
            if self.enable_audio:
                audio_process = (
                    ffmpeg
                    .input(self.input_path)
                    .output(
                        'pipe:',
                        format='f32le',
                        acodec='pcm_f32le',
                        ac=self.channels,
                        ar=self.sample_rate,
                        loglevel='error'
                    )
                    .overwrite_output()
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

                # Start audio playback thread
                audio_thread = Thread(target=self.play_audio, args=(audio_process,), daemon=True)
                audio_thread.start()

            frame_interval = 1.0 / self.target_fps

            while self.running:
                current_time = time.time()
                if current_time - self.last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue

                try:
                    in_bytes = video_process.stdout.read(frame_size)
                    if not in_bytes:
                        time.sleep(0.001)
                        continue

                    frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                    
                    if not self.frame_buffer.full():
                        self.frame_buffer.put(frame)
                        self.last_frame_time = current_time
                    
                except Exception as e:
                    print(f"Error reading frame: {e}")
                    time.sleep(0.001)

        except ffmpeg.Error as e:
            print('FFmpeg error:', e.stderr.decode() if e.stderr else str(e))
        finally:
            video_process.kill()
            if audio_process:
                audio_process.kill()

    def play_audio(self, audio_process):
        if not self.enable_audio:
            return
        self.setup_audio_stream()
        try:
            while self.running:
                audio_chunk = audio_process.stdout.read(self.chunk_size * self.channels * 4)
                if not audio_chunk:
                    break
                
                audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
                self.audio_stream.write(audio_data.tobytes())

        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()

    def process_frames(self):
        while self.running:
            try:
                frame = self.frame_buffer.get(timeout=0.1)
                
                results = self.model(
                    frame,
                    conf=0.5,
                    batch=1,
                    device=self.device,
                    half=True if self.device.startswith("cuda") else False,
                    classes=self.class_id
                )

                for result in results:
                    annotated_frame = result.plot()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    
                    if not self.processed_frames.full():
                        self.processed_frames.put(annotated_frame)

            except Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

    def display_frames(self):
        frame_interval = 1.0 / self.target_fps
        last_display_time = 0

        while self.running:
            try:
                current_time = time.time()
                if current_time - last_display_time < frame_interval:
                    time.sleep(0.001)
                    continue

                frame = self.processed_frames.get(timeout=0.1)
                cv2.imshow('Detection Stream', frame)
                last_display_time = current_time
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    return
            except Empty:
                continue

    def process_video(self):
        self.running = True
        
        read_thread = Thread(target=self.read_stream, daemon=True)
        process_thread = Thread(target=self.process_frames, daemon=True)
        display_thread = Thread(target=self.display_frames, daemon=True)

        read_thread.start()
        process_thread.start()
        display_thread.start()

        try:
            while self.running:
                time.sleep(0.1)
        finally:
            self.cleanup()
            cv2.destroyAllWindows()
            
            read_thread.join(timeout=1.0)
            process_thread.join(timeout=1.0)
            display_thread.join(timeout=1.0)

    def cleanup(self):
        self.running = False
        if self.enable_audio and self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio.terminate()

# Usage - Task: detect, segment, classify, pose. / size = nano, small, medium, large, xlarge
def get_low_latency_stream(streamer):
    session = streamlink.Streamlink()
    options = {
        "twitch-low-latency": True,
        "hls-live-edge": 1,
        "hls-segment-threads": 3
    }

    # Apply options to the session
    for option, value in options.items():
        session.set_option(option, value)
    
    streams = session.streams(f"https://www.twitch.tv/{streamer}")
    print("Available qualities:", streams.keys())
    # Usually: ['audio_only', '160p', '360p', '480p', '720p60', '1080p60', 'worst', 'best']
    if "360p" in streams:
        hls_url = streams['360p'].url
        print(f"Low latency HLS URL for {streamer}: {hls_url}")
        return hls_url
    else:
        print("Desired quality not found.")
        return None

# Example usage
streamer = "nyanners" # username of the stream on Twitch
hls_url = get_low_latency_stream(streamer)

detector = VideoDetector(
    task="segment",
    size="nano",
    input_path=hls_url,
    class_name="person",
    enable_audio=True  # Set to False to disable audio
)
detector.process_video()