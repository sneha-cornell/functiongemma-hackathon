"""Twin Mind — Capture Module

Records or extracts video frames and audio for processing by Gemma 4.
Supports two modes:
  1. Live recording from webcam + microphone
  2. Extraction from an existing video file
"""

import os
import re
import subprocess
import time
import threading
import wave
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARN] sounddevice not installed — no audio. Run: pip install sounddevice")

SAMPLE_RATE = 16000
FRAME_INTERVAL_SEC = 2
CAMERA_WARMUP_FRAMES = 30
FRAME_SIZE = (512, 512)


def create_session_dir(base_dir):
    """Create a timestamped session directory with frames/ subdirectory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session_dir = os.path.join(base_dir, "sessions", timestamp)
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    return session_dir, frames_dir


def save_frame(frame, frames_dir, index):
    """Convert a BGR OpenCV frame to RGB, resize, and save as JPEG."""
    path = os.path.join(frames_dir, f"frame_{index:03d}.jpg")
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize(FRAME_SIZE)
    img.save(path)
    return path


def record_live(duration_sec, session_dir, frames_dir):
    """Record video frames from webcam and audio from microphone simultaneously."""
    print(f"Recording {duration_sec}s of video + audio...\n")

    audio_data = []

    def _record_audio():
        if not AUDIO_AVAILABLE:
            return
        audio_data.append(
            sd.rec(int(duration_sec * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        )
        sd.wait()

    audio_thread = threading.Thread(target=_record_audio)
    audio_thread.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(CAMERA_WARMUP_FRAMES):
        cap.read()
        time.sleep(0.1)
    print("  Camera ready")

    frame_paths = []
    start = time.time()
    last_capture = 0

    while time.time() - start < duration_sec:
        ret, frame = cap.read()
        if not ret:
            continue
        elapsed = time.time() - start
        if elapsed - last_capture >= FRAME_INTERVAL_SEC:
            path = save_frame(frame, frames_dir, len(frame_paths))
            frame_paths.append(path)
            last_capture = elapsed
            print(f"  Frame captured at {elapsed:.1f}s")

    cap.release()
    audio_thread.join()

    audio_path = None
    if AUDIO_AVAILABLE and audio_data:
        audio_path = os.path.join(session_dir, "audio.wav")
        with wave.open(audio_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data[0].tobytes())
        print(f"  Audio saved: {audio_path}")

    print(f"\nCaptured {len(frame_paths)} frames + audio\n")
    return frame_paths, audio_path


def extract_from_video(video_path, session_dir, frames_dir):
    """Extract frames and audio from an existing video file."""
    video_path = os.path.abspath(video_path)
    print(f"Extracting from {video_path}...\n")

    audio_path = os.path.join(session_dir, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16", audio_path],
        check=True, capture_output=True,
    )
    print(f"  Audio extracted: {audio_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval_frames = int(fps * FRAME_INTERVAL_SEC)
    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval_frames == 0:
            path = save_frame(frame, frames_dir, len(frame_paths))
            frame_paths.append(path)
            print(f"  Frame captured at {frame_idx / fps:.1f}s")
        frame_idx += 1

    cap.release()
    print(f"\nExtracted {len(frame_paths)} frames\n")
    return frame_paths, audio_path

