# ==============================================================================
# FINAL, STABLE STREAMLIT APPLICATION
# Filename: streamlit_app.py
# ==============================================================================
import streamlit as st
import os
import tempfile
import numpy as np
import cv2
from ultralytics import YOLO
import tracking_utils
import ui
from advanced_analysis import run_advanced_analysis
from video_processing import trim_video_ffmpeg
import config # Import config directly as a module

st.set_page_config(layout="wide", page_title="Football Analysis App")

@st.cache_resource
def get_yolo_model() -> YOLO: return YOLO(config.YOLO_MODEL_WEIGHTS)

def main():
    st.title("⚽ Football Match Analysis App")
    for key in ['temp_video_path', 'homography_matrix', 'analysis_complete', 'homography_points', 'trimmed_video_path']:
        if key not in st.session_state: st.session_state[key] = None

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

        # Define the original full video path (hardcoded as per your request)
        # IMPORTANT: This path must point to your *source* video file for trimming.
        full_video_path = r"C:\Users\daadi\ProjectsFootballAnalysis\SoccerNetData\england_epl\2014-2015\2015-04-11 - 19-30 Burnley 0 - 1 Arsenal\1_224p.mkv"
        
        # Define output path for trimmed videos
        trimmed_output_dir = os.path.join(config.PROJECT_ROOT, "trimmed_videos")
        os.makedirs(trimmed_output_dir, exist_ok=True)
        trimmed_output_filename = "trimmed_video.mp4" # You can make this dynamic if needed
        trimmed_output_path = os.path.join(trimmed_output_dir, trimmed_output_filename)


        if uploaded_file and st.session_state.temp_video_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue()); st.session_state.temp_video_path = tmp.name

            # Reset trimmed_video_path if a new file is uploaded
            st.session_state.trimmed_video_path = None

            # Load cached homography for the newly uploaded video
            cached_matrix = tracking_utils.load_homography_from_cache(config.HOMOGRAPHY_CACHE_PATH, st.session_state.temp_video_path)
            if cached_matrix is not None:
                st.session_state.homography_matrix = cached_matrix; st.success("✅ Cached homography loaded!")
            st.rerun()

        # Determine which video path to use for analysis and display across all tabs
        video_for_analysis_path = st.session_state.temp_video_path # Default to the uploaded video
        if st.session_state.trimmed_video_path:
            video_for_analysis_path = st.session_state.trimmed_video_path # Override if a trimmed video exists and was successfully created

        if st.session_state.temp_video_path: # Only show controls if a video is uploaded
            st.subheader("Video Trimming (FFmpeg)")
            st.write(f"Source Video for Trimming: `{full_video_path}`")
            st.write(f"FFmpeg Executable Path: `{config.FFMPEG_PATH}`")

            trim_start_time = st.text_input("Start Time (HH:MM:SS)", "00:00:00", key="trim_start_time")
            # IMPORTANT: Guide user on FFmpeg duration format (no 60 seconds)
            trim_duration = st.text_input("Duration (HH:MM:SS, e.g., 00:01:00 for 1 min)", "00:00:30", key="trim_duration")
            st.info("💡 For durations like 60s, use `00:01:00` (1 minute), not `00:00:60`.")

            if st.button("Trim Video", key="trim_button"):
                with st.spinner(f"Trimming video from {trim_start_time} for {trim_duration}..."):
                    success = trim_video_ffmpeg(full_video_path, trimmed_output_path, trim_start_time, trim_duration, config.FFMPEG_PATH)
                    if success:
                        st.session_state.trimmed_video_path = trimmed_output_path
                        st.success(f"✅ Video trimmed successfully to: `{trimmed_output_path}`")
                        # After successful trim, update video_for_analysis_path to the new trimmed one
                        video_for_analysis_path = st.session_state.trimmed_video_path # Update it here as well
                        st.experimental_rerun() # Re-run to update the displayed video and ensure path is recognized
                    else:
                        st.error("Failed to trim video. Check console for FFmpeg errors.")

            # Display the video that is currently set for analysis
            if video_for_analysis_path:
                st.video(video_for_analysis_path, format="video/mp4", start_time=0)
                st.info(f"**Video currently selected for analysis:** `{video_for_analysis_path}`")
            else:
                st.warning("No video selected or trimmed for analysis yet.")


            st.subheader("Homography Control")
            if st.button("Calculate Homography from Points", key="calc_homography_button"):
                if st.session_state.get('homography_points') and len(st.session_state.homography_points) == 4:
                    source_points = np.array(st.session_state.homography_points, dtype=np.float32)
                    destination_points = config.PITCH_REAL_WORLD_COORDINATES_METERS
                    h_matrix, _ = cv2.findHomography(source_points, destination_points)
                    st.session_state['homography_matrix'] = h_matrix
                    # Save homography associated with the specific video path being analyzed
                    tracking_utils.save_homography_to_cache(config.HOMOGRAPHY_CACHE_PATH, video_for_analysis_path, h_matrix)
                    st.success("✅ Homography matrix calculated and saved!")
                else:
                    st.error("Please select exactly 4 points on the Homography Setup tab first.")

            st.subheader("Analysis Control")
            if st.button("🚀 Start Analysis", use_container_width=True, type="primary", key="start_analysis_button"):
                if not video_for_analysis_path:
                    st.error("No video available for analysis. Please upload or trim a video first.")
                    return

                with st.status("Running analysis...", expanded=True) as status:
                    model, h_matrix = get_yolo_model(), st.session_state.get('homography_matrix')
                    if h_matrix is None: st.warning("Homography not set. Analysis will run in pixel coordinates.", icon="⚠️")
                    status.update(label="Processing video...")

                    results = run_advanced_analysis(video_path=video_for_analysis_path, model=model, homography_matrix=h_matrix)
                    st.session_state.update(results); st.session_state['analysis_complete'] = True
                    status.update(label="✅ Analysis Complete!", state="complete"); st.success("Analysis complete!")
                    st.rerun() # Force rerun to update tabs with results

    # --- Main Page Display Logic ---
    # Only display tabs if a video path is available for analysis
    if video_for_analysis_path:
        ui.display_results_tabs(st.session_state, video_for_analysis_path) # <<< Corrected call
    else:
        st.info("👋 Welcome! Please upload a video using the sidebar to begin.")

if __name__ == "__main__":
    main()