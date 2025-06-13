# ==============================================================================
# FINAL ADVANCED ANALYSIS MODULE
# Filename: advanced_analysis.py
# ==============================================================================
import numpy as np
import pandas as pd
from enum import Enum
import supervision as sv
from ultralytics import YOLO
import cv2
from config import *
from tracking_utils import yield_video_frames
from visualization import draw_annotations
import os
import yaml

def get_bbox_center(bbox):
    """
    Calculate the center of a bounding box (x_min, y_min, x_max, y_max).
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2, (y_min + y_max) / 2

class PassState(Enum):
    NO_POSSESSION, POSSESSION, BALL_IN_FLIGHT = 1, 2, 3

class PassDetector:
    def __init__(self):
        self.state = PassState.NO_POSSESSION
        self.player_in_possession = None
        self.passer = None
        self.pass_start_position = None
        self.pass_start_frame = None
        self.frames_in_flight = 0
        self.detected_passes = []

    def _get_player_closest_to_ball(self, ball_pos, players_pos):
        if not players_pos or ball_pos is None: return None, float('inf')
        player_ids, positions = list(players_pos.keys()), np.array(list(players_pos.values()))
        distances = np.linalg.norm(positions - np.array(ball_pos), axis=1)
        min_idx = np.argmin(distances)
        return player_ids[min_idx], distances[min_idx]

    def update(self, frame_number, ball_pos, players_pos):
        closest_player_id, distance_to_ball = self._get_player_closest_to_ball(ball_pos, players_pos)

        # DEBUG PRINTS FOR PASS DETECTION STATE - UNCOMMENTED
        print(f"Frame {frame_number}: State={self.state.name}, Closest Player={closest_player_id}, Dist to Ball={distance_to_ball:.2f}")

        if self.state == PassState.NO_POSSESSION:
            if distance_to_ball < POSSESSION_THRESHOLD_METERS:
                self.state, self.player_in_possession = PassState.POSSESSION, closest_player_id
                print(f"  -> NEW STATE: POSSESSION by {closest_player_id}")
        elif self.state == PassState.POSSESSION:
            if self.player_in_possession != closest_player_id or distance_to_ball > POSSESSION_THRESHOLD_METERS:
                self.passer = self.player_in_possession
                self.pass_start_position = players_pos.get(self.player_in_possession, ball_pos)
                self.pass_start_frame = frame_number
                self.frames_in_flight = 0
                self.state = PassState.BALL_IN_FLIGHT
                print(f"  -> NEW STATE: BALL_IN_FLIGHT (Passer: {self.passer})")
        elif self.state == PassState.BALL_IN_FLIGHT:
            self.frames_in_flight += 1
            if distance_to_ball < POSSESSION_THRESHOLD_METERS:
                if (receiver := closest_player_id) != self.passer:
                    self.detected_passes.append({'passer_id': self.passer, 'receiver_id': receiver, 'start_frame': self.pass_start_frame, 'end_frame': frame_number, 'x_start': self.pass_start_position[0], 'y_start': self.pass_start_position[1], 'x_end': ball_pos[0], 'y_end': ball_pos[1]})
                    print(f"  -> PASS DETECTED: {self.passer} to {receiver}")
                self.state, self.player_in_possession = PassState.POSSESSION, receiver
                print(f"  -> NEW STATE: POSSESSION by {receiver}")
            elif self.frames_in_flight > PASS_TIME_WINDOW_FRAMES:
                self.state = PassState.NO_POSSESSION
                print(f"  -> NEW STATE: NO_POSSESSION (Pass timed out)")

def calculate_player_stats(tracker_data: pd.DataFrame) -> dict:
    if tracker_data.empty: return {}
    player_stats = {}
    for player_id, player_df in tracker_data.groupby('track_id'):
        coords = player_df.sort_values(by='frame')[['x_transformed', 'y_transformed']].to_numpy()
        total_distance = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
        duration_frames = player_df['frame'].max() - player_df['frame'].min()
        average_speed = (total_distance / (duration_frames / FRAME_RATE)) if duration_frames > 0 else 0
        player_stats[player_id] = {'total_distance_meters': total_distance, 'average_speed_mps': average_speed}
    return player_stats

def run_advanced_analysis(video_path: str, model: YOLO, homography_matrix: np.ndarray):
    print(f"YOLO Model Classes: {model.names}")

    # Define the path to your custom ByteTrack configuration
    BYTETRACK_CONFIG_PATH = 'custom_bytetrack.yaml'

    # Load ByteTrack configuration if the file exists
    track_activation_threshold = 0.4
    lost_track_buffer = 30
    minimum_matching_threshold = 0.85

    if os.path.exists(BYTETRACK_CONFIG_PATH):
        try:
            with open(BYTETRACK_CONFIG_PATH, 'r') as f:
                bytetrack_config = yaml.safe_load(f)

            track_activation_threshold = bytetrack_config.get('track_activation_threshold', track_activation_threshold)
            lost_track_buffer = bytetrack_config.get('lost_track_buffer', lost_track_buffer)
            minimum_matching_threshold = bytetrack_config.get('minimum_matching_threshold', minimum_matching_threshold)
            print(f"Loaded ByteTrack config: track_activation_threshold={track_activation_threshold}, lost_track_buffer={lost_track_buffer}, minimum_matching_threshold={minimum_matching_threshold}")
        except Exception as e:
            print(f"Error loading ByteTrack config from {BYTETRACK_CONFIG_PATH}: {e}. Using default parameters.")

    tracker = sv.ByteTrack(
        frame_rate=FRAME_RATE,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold
    )

    pass_detector, tracker_data_list, annotated_frames = PassDetector(), [], []
    for frame_idx, frame in enumerate(yield_video_frames(video_path)):
        detections = sv.Detections.from_ultralytics(model(frame, verbose=False)[0])
        tracked_objects = tracker.update_with_detections(detections)

        # DEBUGGING FOR BALL DETECTION
        if frame_idx < 10 or frame_idx % 100 == 0:
            detected_classes = detections.class_id if detections.class_id is not None else []
            sports_ball_detected = CLASS_ID_SPORTS_BALL in detected_classes
            print(f"[DEBUG] Frame {frame_idx}: Detections count: {len(detections.xyxy)}, Sports ball detected: {sports_ball_detected}")

        annotated_frames.append(draw_annotations(frame.copy(), tracked_objects))

        current_frame_players, current_frame_ball_pos = {}, None
        if homography_matrix is not None:
            for bbox, _, class_id, track_id in zip(tracked_objects.xyxy, tracked_objects.confidence, tracked_objects.class_id, tracked_objects.tracker_id):
                bbox_center = get_bbox_center(bbox)
                tp = cv2.perspectiveTransform(np.array([[bbox_center]], dtype=np.float32), homography_matrix)[0][0]

                if class_id == CLASS_ID_PERSON:
                    current_frame_players[track_id] = tuple(tp)
                    tracker_data_list.append({
                        'frame': frame_idx,
                        'track_id': track_id,
                        'x_transformed': tp[0],
                        'y_transformed': tp[1]
                    })
                elif class_id == CLASS_ID_SPORTS_BALL:
                    current_frame_ball_pos = tuple(tp)
                    tracker_data_list.append({
                        'frame': frame_idx,
                        'track_id': -99,
                        'x_transformed': tp[0],
                        'y_transformed': tp[1]
                    })

        pass_detector.update(frame_idx, current_frame_ball_pos, current_frame_players)

        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}: Detected {len(pass_detector.detected_passes)} passes so far.")

    tracker_data_df = pd.DataFrame(tracker_data_list)
    return {
        'tracker_data': tracker_data_df,
        'pass_events': pd.DataFrame(pass_detector.detected_passes),
        'player_stats': calculate_player_stats(tracker_data_df),
        'annotated_frames': annotated_frames
    }