# ==============================================================================
# FINAL, STABLE VISUALIZATION MODULE
# Filename: visualization.py
# ==============================================================================
import cv2
import numpy as np
import supervision as sv
from config import *

def draw_annotations(frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
    """
    Draws bounding boxes and labels for all tracked objects on a single frame.
    """
    if detections is None or len(detections) == 0:
        return frame

    annotated_frame = frame.copy()
    
    # Setup annotators for drawing
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=0.5, text_color=sv.Color.WHITE)

    # --- START OF FIX ---
    # Create labels using a robust list comprehension that handles missing tracker_id.
    # We iterate through the main properties of the 'detections' object.
    
    labels = []
    # The 'tracker_id' might be None for some detections, so we provide a default 'N/A'.
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id if detections.tracker_id is not None else [None] * len(detections)):
        class_name = "Ball" if class_id == CLASS_ID_SPORTS_BALL else "Player"
        
        if tracker_id is not None:
            labels.append(f"#{tracker_id} {class_name}")
        else:
            labels.append(class_name) # Label without ID if not yet tracked
    # --- END OF FIX ---

    # Annotate the frame with the generated boxes and labels
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame