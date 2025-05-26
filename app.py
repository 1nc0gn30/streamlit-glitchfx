import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import random

# --- Effect imports ---
from basic_datamosh import datamosh_effect as basic_effect
from advanced_datamosh import advanced_datamosh_effect as advanced_effect
from glitch_effect_stack import add_glitch_effect as glitch_stack_effect
from frame_overlay_mosher import add_glitch_effect as overlay_effect
from fake_detection_glitch import datamosh_effect_1 as detection_1, datamosh_effect_2 as detection_2, fake_detection_overlay
from clip_glitcher_simple import datamosh_effect_1 as clip_1, datamosh_effect_2 as clip_2
from video_mashup_blender import advanced_mash_effect

# --- Effect definitions ---
EFFECTS = {
    "Basic Datamosh": basic_effect,
    "Advanced Datamosh": advanced_effect,
    "Glitch Stack": glitch_stack_effect,
    "Frame Overlay Mosher": overlay_effect,
    "Clip Glitcher (Moshes)": lambda f: clip_2(clip_1(f)),
    "Fake Detection Glitch": lambda f: fake_detection_overlay(detection_2(detection_1(f))),
}

SPECIAL_EFFECTS = {
    "Video Mashup Blender (2 videos)": advanced_mash_effect,
}

# --- Core processor ---
def apply_effect(input_path, output_path, effect_func):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = effect_func(frame)
        out.write(frame)

    cap.release()
    out.release()

# --- Dual video processor ---
def apply_dual_effect(video1_bytes, video2_bytes, output_path, effect_func):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f2:
        f1.write(video1_bytes.read())
        f2.write(video2_bytes.read())
        f1.flush()
        f2.flush()

        cap1 = cv2.VideoCapture(f1.name)
        cap2 = cv2.VideoCapture(f2.name)

        if not cap1.isOpened() or not cap2.isOpened():
            st.error("Failed to open one or both videos.")
            return

        fps = cap1.get(cv2.CAP_PROP_FPS)
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            frame2 = cv2.resize(frame2, (width, height))
            out.write(effect_func(frame1, frame2))

        cap1.release()
        cap2.release()
        out.release()

# --- Streamlit UI ---
st.set_page_config(page_title="Glitch Video Lab", layout="centered")
st.title("🎛️ Glitch Video Lab — Advanced Mode")
st.caption("Upload one or two videos, apply glitch/detection/mash effects, and download the result.")

effect_type = st.selectbox("🎨 Choose Effect Type", list(EFFECTS.keys()) + list(SPECIAL_EFFECTS.keys()))

if effect_type in EFFECTS:
    video_file = st.file_uploader("📤 Upload MP4 video", type=["mp4"])
    if video_file and st.button("⚡ Apply Glitch"):
        with st.spinner("Processing..."):
            temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_in.write(video_file.read())
            temp_in.close()

            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_out.close()

            try:
                apply_effect(temp_in.name, temp_out.name, EFFECTS[effect_type])
                st.success("✅ Glitch complete!")
                with open(temp_out.name, "rb") as f:
                    st.download_button("📥 Download", f, file_name=f"{effect_type.lower().replace(' ', '_')}.mp4")
            except Exception as e:
                st.error(f"❌ Error: {e}")

elif effect_type in SPECIAL_EFFECTS:
    col1, col2 = st.columns(2)
    with col1:
        vid1 = st.file_uploader("📼 Video 1", type=["mp4"], key="vid1")
    with col2:
        vid2 = st.file_uploader("📼 Video 2", type=["mp4"], key="vid2")

    if vid1 and vid2 and st.button("🎬 Mash 'em"):
        with st.spinner("Combining videos..."):
            output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output.close()
            try:
                apply_dual_effect(vid1, vid2, output.name, SPECIAL_EFFECTS[effect_type])
                st.success("✅ Mash complete!")
                with open(output.name, "rb") as f:
                    st.download_button("📥 Download Mashup", f, file_name="mashed_output.mp4")
            except Exception as e:
                st.error(f"❌ Error: {e}")

