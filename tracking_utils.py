# ==============================================================================
# FINAL MEMORY-EFFICIENT TRACKING UTILITIES
# Filename: tracking_utils.py
# ==============================================================================
import cv2
import numpy as np
import os
import json
from typing import Optional

def save_homography_to_cache(cache_path: str, key: str, matrix: np.ndarray):
    try:
        cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        cache[key] = matrix.tolist()
        with open(cache_path, 'w') as f: json.dump(cache, f, indent=4)
    except Exception as e: print(f"Error saving homography to cache: {e}")

def load_homography_from_cache(cache_path: str, key: str) -> Optional[np.ndarray]:
    if not os.path.exists(cache_path): return None
    try:
        with open(cache_path, 'r') as f: cache = json.load(f)
        return np.array(cache[key], dtype=np.float32) if key in cache else None
    except Exception as e:
        print(f"Error loading homography from cache: {e}")
        return None

def yield_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video file: {video_path}")
    while True:
        success, frame = cap.read()
        if not success: break
        yield frame
    cap.release()

def get_video_properties(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0.0, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, width, height

def get_frame_at_index(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    cap.release()
    return frame if success else None