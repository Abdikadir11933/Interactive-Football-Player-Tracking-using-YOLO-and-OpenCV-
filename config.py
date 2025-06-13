# ==============================================================================
# FINAL CONFIGURATION FILE
# Filename: config.py
# ==============================================================================
import os
import numpy as np

# --- Project Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
HOMOGRAPHY_CACHE_PATH = os.path.join(PROJECT_ROOT, 'cached_homographies.json')
os.makedirs(MODELS_DIR, exist_ok=True)

# Define your FFMPEG_PATH here
# IMPORTANT: Double-check this path. If ffmpeg.exe is not exactly here, it will fail.
FFMPEG_PATH = r"C:\Users\daadi\ProjectsFootballAnalysis\ffmpeg_extracted\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# --- Model Settings ---
# Upgraded to the 'large' model for potentially highest accuracy on small objects like the ball.
YOLO_MODEL_WEIGHTS = 'yolov8l.pt'

# --- Homography Settings ---
# Real-world dimensions (in meters) of a standard penalty box
PITCH_REAL_WORLD_COORDINATES_METERS = np.array([
    [0.0, 0.0], [40.32, 0.0], [40.32, 16.5], [0.0, 16.5]
], dtype=np.float32)

# --- Visualization Settings ---
PITCH_BACKGROUND_COLOR = '#0f250f' # A slightly darker, richer green
PITCH_LINE_COLOR = '#cccccc'       # A light grey for contrast
HEATMAP_COLORMAP = 'hot'           # Options: 'hot', 'viridis', 'plasma', 'magma'

# --- Analysis Parameters ---
CLASS_ID_PERSON = 0
CLASS_ID_SPORTS_BALL = 32
FRAME_RATE = 30
# Increased for more lenient possession detection (adjust based on video scale)
POSSESSION_THRESHOLD_METERS = 7.0
# Increased for more lenient pass detection time window (adjust based on game speed)
PASS_TIME_WINDOW_FRAMES = 150