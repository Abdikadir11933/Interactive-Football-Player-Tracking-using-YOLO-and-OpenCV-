# ==============================================================================
# FINAL, STABLE, AND COMPLETE UI MODULE
# Filename: ui.py
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
from scipy.ndimage import gaussian_filter
from streamlit_drawable_canvas import st_canvas
import config
import tracking_utils

def display_results_tabs(state, video_path): # <<< ADDED video_path parameter
    """
    Creates the main tab layout and calls the specific function for each tab.
    'state' is st.session_state from the main app.
    'video_path' is the path to the video currently being analyzed (original or trimmed).
    """
    tab_names = ["📋 Overview", "🛠️ Homography Setup", "🖼️ Frame Viewer", "🔥 Heatmaps", "📡 Pass Network", "📈 Player Stats"]
    tabs = st.tabs(tab_names)
    with tabs[0]: display_overview_tab()
    with tabs[1]: display_homography_setup_tab(state, video_path) # <<< PASSING video_path here
    with tabs[2]: display_frame_viewer_tab(state)
    with tabs[3]: display_heatmap_tab(state)
    with tabs[4]: display_pass_network_tab(state)
    with tabs[5]: display_player_stats_tab(state)

def display_overview_tab():
    st.header("📋 Project Overview")
    st.markdown("""
    Welcome! This is the final, stable version of the app.
    1.  **Upload Video** in the sidebar.
    2.  Go to **Homography Setup** to select your 4 points on the canvas.
    3.  Click **'Calculate Homography'** in the sidebar to confirm the points.
    4.  Click **Start Analysis** to process the video.
    5.  Explore the results in the other tabs.
    """)

def display_homography_setup_tab(state, video_path): # <<< ADDED video_path parameter
    """
    This final version uses the 'ghost image' workaround for a stable and
    user-friendly experience, allowing for precise point selection.
    """
    st.header("🛠️ Homography Setup")

    # Use the passed video_path argument for checks and operations
    if not video_path:
        st.warning("Please upload a video first to enable homography setup.")
        return

    _, frame_count, _, _ = tracking_utils.get_video_properties(video_path)
    if frame_count == 0:
        st.error("Could not read video properties.")
        return

    st.markdown("""
    **Instructions:** Select a clear frame using the slider. A reference image will appear. Click the corresponding points for the penalty box on the interactive canvas below, which has a faint guide.
    """)

    frame_idx = st.slider("Select Frame for Calibration", 0, frame_count - 1, 0, key="homography_slider")

    raw_frame = tracking_utils.get_frame_at_index(video_path, frame_idx)

    if raw_frame is not None:
        # Create a full-quality reference image for the user to see clearly.
        reference_image = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
        st.image(reference_image, caption="Full-Quality Reference Frame")
        st.markdown("---")

        # Create a faint, "ghost" version of the frame for the canvas background.
        faded_background = (raw_frame * 0.4 + np.full_like(raw_frame, 255) * 0.6).astype(np.uint8)
        faded_pil_image = Image.fromarray(cv2.cvtColor(faded_background, cv2.COLOR_BGR2RGB))

        # Display the interactive canvas with the faint background guide.
        st.write("Click on the canvas below to set points:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#e00",
            background_image=faded_pil_image,
            update_streamlit=True,
            height=reference_image.height,
            width=reference_image.width,
            drawing_mode="point",
            point_display_radius=8,
            key=f"homography_canvas_{frame_idx}"
        )

        if canvas_result.json_data and "objects" in canvas_result.json_data:
            points = [(p['left'], p['top']) for p in canvas_result.json_data["objects"]]
            st.session_state['homography_points'] = points

            st.write(f"**{len(points)} / 4 points selected.**")
            st.dataframe(pd.DataFrame(points, columns=['x', 'y']))

            if len(points) == 4:
                st.info("Points selected. Go to the sidebar and click 'Calculate Homography'.")
            elif len(points) > 4:
                st.error("Too many points! Please click the trash icon on the canvas to reset.")
    else:
        st.error(f"Could not retrieve frame {frame_idx}.")


def display_frame_viewer_tab(state):
    st.header("🖼️ Frame Viewer")
    if 'annotated_frames' not in state or not state['annotated_frames']:
        st.info("Run analysis to generate tracked frames.")
        return

    frames_to_show = state['annotated_frames']
    frame_idx = st.slider("Select Annotated Frame", 0, len(frames_to_show) - 1, 0, key="viewer_slider")

    if frames_to_show:
        st.image(frames_to_show[frame_idx], channels="BGR", caption=f"Annotated Frame {frame_idx}")


def display_heatmap_tab(state):
    st.header("🔥 Player Activity Heatmap")
    st.markdown("Visualize where players spent most of their time on the pitch.")

    if 'tracker_data' not in state or state['tracker_data'].empty:
        st.info("No heatmap data available. Please run the analysis first.")
        return

    tracker_df = state['tracker_data']
    all_player_ids = sorted([pid for pid in tracker_df['track_id'].unique() if pid is not None and pid != -99]) # Exclude ball ID
    
    if not all_player_ids:
        st.warning("No players were tracked in the analysis (or all are the ball).")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_player_id = st.selectbox("Select Player ID:", options=all_player_ids)
    with col2:
        heatmap_style = st.selectbox("Heatmap Style:", ["Density Plot (Smoothed)", "Contour Plot", "Both"])

    if not selected_player_id:
        return

    player_df = tracker_df[tracker_df['track_id'] == selected_player_id]

    pitch = VerticalPitch(
        pitch_type='statsbomb',
        pitch_color=config.PITCH_BACKGROUND_COLOR,
        line_color=config.PITCH_LINE_COLOR,
        linewidth=2,
        stripe=False
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.set_facecolor(config.PITCH_BACKGROUND_COLOR)

    bin_statistic = pitch.bin_statistic(
        player_df.x_transformed,
        player_df.y_transformed,
        statistic='count',
        bins=(40, 40)
    )

    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], sigma=1.8)

    if heatmap_style == "Density Plot (Smoothed)" or heatmap_style == "Both":
        pitch.heatmap(
            bin_statistic,
            ax=ax,
            cmap=config.HEATMAP_COLORMAP,
            alpha=0.7
        )

    if heatmap_style == "Contour Plot" or heatmap_style == "Both":
        levels = np.linspace(bin_statistic['statistic'].min(), bin_statistic['statistic'].max(), 10)
        pitch.contour(
            bin_statistic,
            ax=ax,
            cmap=config.HEATMAP_COLORMAP,
            levels=levels,
            alpha=0.8
        )

    ax.set_title(f"Activity Heatmap for Player {int(selected_player_id)}", color='white', fontsize=18)

    if ax.collections:
        cbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.7, aspect=10)
        cbar.set_label('Activity Count (Smoothed)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    st.pyplot(fig, transparent=True)

def display_pass_network_tab(state):
    st.header("📡 Tactical Pass Network")
    if 'pass_events' not in state or state['pass_events'].empty:
        st.info("No pass events detected."); return

    passes, tracks = state['pass_events'], state['tracker_data']
    avg_locs = tracks.groupby('track_id').agg(x=('x_transformed', 'mean'), y=('y_transformed', 'mean')).reset_index()
    passer_counts = passes.groupby('passer_id').size().reset_index(name='pass_count')
    pairs = passes.groupby(['passer_id', 'receiver_id']).size().reset_index(name='pair_count')

    if pairs.empty:
        st.info("Not enough pass interactions to create a network."); return

    pairs = pairs.merge(avg_locs, left_on='passer_id', right_on='track_id').merge(avg_locs.rename(columns={'track_id': 'receiver_id', 'x': 'x_end', 'y': 'y_end'}), on='receiver_id')
    avg_locs = avg_locs.merge(passer_counts, left_on='track_id', right_on='passer_id', how='left').fillna(0)

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color=config.PITCH_BACKGROUND_COLOR, line_color=config.PITCH_LINE_COLOR)
    fig, ax = pitch.draw(figsize=(16, 11)); fig.set_facecolor(config.PITCH_BACKGROUND_COLOR)

    pair_alpha = (pairs.pair_count / pairs.pair_count.max()) * 0.8 + 0.2 if not pairs.empty and pairs.pair_count.max() > 0 else 0.5
    node_size = (avg_locs.pass_count / avg_locs.pass_count.max()) * 1200 + 300 if not avg_locs.empty and avg_locs.pass_count.max() > 0 else 300

    pitch.arrows(pairs.x, pairs.y, pairs.x_end, pairs.y_end, ax=ax, width=2, headwidth=10, color='white', alpha=pair_alpha)
    pitch.scatter(avg_locs.x, avg_locs.y, s=node_size, ax=ax, color='red', edgecolors='white', lw=2, zorder=2)

    for _, row in avg_locs.iterrows():
        pitch.text(row.x, row.y, str(int(row.track_id)), ax=ax, color='white', va='center', ha='center', zorder=3)

    ax.set_title("Tactical Pass Network", color='white'); st.pyplot(fig, transparent=True)

def display_player_stats_tab(state):
    st.header("📈 Player Statistics")
    if 'player_stats' not in state or not state['player_stats']:
        st.info("Run analysis to generate stats."); return

    stats = state['player_stats']
    df_data = [{"Player ID": int(pid), "Avg Speed (m/s)": s.get('average_speed_mps', 0), "Distance (m)": s.get('total_distance_meters', 0)} for pid, s in stats.items()]

    if not df_data:
        st.info("No player stats to display."); return

    st.dataframe(pd.DataFrame(df_data).set_index("Player ID").round(2))