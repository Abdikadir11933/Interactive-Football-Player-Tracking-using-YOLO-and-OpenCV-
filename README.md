Football Match Analysis App
An advanced computer vision application built with Python and Streamlit to perform object detection, player tracking, and tactical analysis on football match videos.

Key Features
This application processes football match footage to provide a rich set of data visualizations and tactical insights:

Object Detection & Tracking: Utilizes a YOLOv8 model to detect players, referees, and the ball, and tracks them throughout the video using ByteTrack.
Homography Transformation: Allows users to map player positions from 2D video pixels to real-world 3D pitch coordinates (in meters) for accurate analysis.
Camera Motion Compensation: Estimates and compensates for camera movement to stabilize player tracks.
Automated Team Assignment: Assigns players to teams using either AI-based visual embeddings (Siglip) or dominant jersey color clustering.
Player Statistics: Calculates and displays key performance metrics for each player, including total distance covered and average speed.
Tactical Visualizations:
Player Heatmaps: Generates heatmaps to show player positioning and areas of influence.
Pass Network: Creates a network graph visualizing the frequency and average location of passes between players.
Interactive UI: A multi-tab Streamlit interface allows for easy video upload, configuration, and exploration of all analysis results.
Project Structure
football-analysis-project/
│
├── data/
│   └── models/
│       └── best.pt               # Your YOLOv8 model weights
│
├── football_final_env/           # Your Conda environment
│
├── app/
│   ├── streamlit_app.py          # Main application file
│   ├── tracker.py                # Core class for detection and tracking
│   ├── ui.py                     # Functions for all UI tabs
│   ├── analysis.py               # Speed, distance, and pass detection logic
│   ├── team_assigner.py          # Team assignment (color & Siglip)
│   ├── tracking_utils.py         # Helper functions (video I/O, Kalman filter)
│   ├── camera_movement_estimator.py # Camera motion logic
│   ├── player_assigner_ball.py   # Ball possession logic
│   └── config.py                 # Central configuration for all parameters
│
├── custom_bytetrack.yaml         # Configuration for the ByteTrack algorithm
└── README.md                     # This file
Setup and Installation
Follow these steps to set up the project environment and run the application.

Prerequisites
Conda installed on your system.
(Optional) An NVIDIA GPU with CUDA drivers installed for faster processing.
1. Create and Activate the Conda Environment
This project requires specific library versions for stability. Create a dedicated environment for it.

Bash

# Create a new environment with Python 3.10
conda create --name football_final_env python=3.10

# Activate the new environment
conda activate football_final_env
2. Install All Dependencies
Run this single command in your activated environment to install all the required libraries with the correct, stable versions.

Bash

pip install streamlit==1.35.0 ultralytics==8.2.2 "supervision==0.22.0" opencv-python==4.9.0.80 pandas==2.2.2 numpy==1.26.4 PyYAML==6.0.1 scikit-learn==1.4.2 umap-learn==0.5.6 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 transformers==4.41.2 more-itertools==10.3.0 filterpy==1.4.5 mplsoccer==1.4.0 streamlit-drawable-canvas==0.9.3 sentencepiece==0.2.0 matplotlib==3.8.4 tqdm==4.66.4
3. Place Your Model
Place your trained YOLOv8 model file (e.g., best.pt) inside the models/ directory.

Configuration
The application's behavior can be tuned via two main configuration files:

config.py: The central hub for all settings. Here you can change:

YOLO_MODEL_WEIGHTS: The path to the model file.
PLAYER_CLASS_NAME: The exact class name for players in your model (e.g., "person").
DETECTION_CONF_THRESHOLD: The confidence required to consider a detection valid.
And many other parameters for colors, distances, and thresholds.
custom_bytetrack.yaml: Contains parameters specifically for the ByteTrack algorithm.

track_thresh: The confidence score required to initialize a new track.
track_buffer: The number of frames to keep a lost track before deleting it.
How to Run the Application
Activate the Conda environment:
Bash

conda activate football_final_env
Navigate to the project's root directory in your terminal.
Run the Streamlit app:
Bash

streamlit run streamlit_app.py
The application will open in your default web browser.

How to Use the App
Upload Video: Use the file uploader in the sidebar to select a football match video.
Set Homography (Recommended): Go to the "🛠️ Homography Setup" tab. Select a clear frame, click on the four corners of the pitch, and click "Calculate and Save Homography". This enables real-world metric calculations.
Run Analysis: Go back to the sidebar and click the "🚀 Start Analysis" button.
Explore Results: Once the pipeline is complete, navigate through the various tabs (Player Stats, Heatmaps, Pass Network, etc.) to view the analysis.
Modules Overview
File	Description
streamlit_app.py	The main application file. It handles the UI sidebar, orchestrates the analysis pipeline, and manages state.
tracker.py	Contains the core Tracker class responsible for object detection and tracking.
ui.py	Defines the functions that create and display all the data visualization tabs.
config.py	A centralized file for all project constants and configuration parameters.
analysis.py	Contains classes like SpeedAndDistanceEstimator and PassDetector for calculating metrics.
team_assigner.py	Handles assigning players to teams using either color clustering or Siglip visual embeddings.
tracking_utils.py	A collection of helper functions for tasks like video I/O, Kalman filtering, and homography.
camera_movement_estimator.py	Estimates and corrects for camera motion between frames to stabilize object positions.
player_assigner_ball.py	A simple utility to determine which player is in possession of the ball.

Vie Sheetsiinbeing ashes
License
This project is licensed under the MIT License. See the LICENSE file for details.
