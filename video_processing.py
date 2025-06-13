import subprocess
import os
import streamlit as st

@st.cache_resource
def get_ffmpeg_path(path: str) -> str:
    if not os.path.exists(path):
        st.error(f"FFmpeg not found at: {path}")
        return ""
    return path

def trim_video_ffmpeg(input_path: str, output_path: str, start_time: str, duration: str, ffmpeg_path: str) -> bool:
    ffmpeg_exec = get_ffmpeg_path(ffmpeg_path)
    if not ffmpeg_exec:
        return False
    command = [ffmpeg_exec, "-ss", start_time, "-i", input_path, "-t", duration, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-y", output_path]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _, stderr = process.communicate(timeout=300)
        if process.returncode != 0:
            st.error("FFmpeg failed:")
            st.text(stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        st.error("FFmpeg timed out")
        return False