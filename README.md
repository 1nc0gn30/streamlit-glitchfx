# 🎛️ Glitch Video Lab

**Glitch Video Lab** is a browser-based tool built with [Streamlit](https://streamlit.io) for applying advanced glitch and datamosh effects to your videos using Python, OpenCV, and MoviePy.

🎥 Upload your `.mp4`  
🎨 Choose your glitch flavor  
⚡ Click one button  
📥 Download the chaos  

---

## 🚀 Features

- 🔨 Basic & Advanced Datamoshing
- 🌈 RGB Glitch Stacks & Pixel Noise
- 📦 Fake Object Detection Overlays
- 🌀 Dual-video Mashups (frame blending madness)
- 🎬 MoviePy FX: invert, mirror, speed-warp
- 🖥️ 100% Browser UI (no editing software needed)
- 💥 Instant download after glitching

---

## 🖼️ Interface

![Screenshot of Glitch Video Lab](https://via.placeholder.com/1000x500?text=Glitch+Video+Lab+Preview)

---

## 🧰 Requirements

Install with:

```bash
pip install -r requirements.txt

Core Dependencies:

    streamlit

    opencv-python

    numpy

    moviepy

    ffmpeg-python

🧪 Usage

Run locally:

streamlit run app.py

Then open your browser to http://localhost:8501.
📂 File Structure

📦 glitch-video-lab/
├── app.py                   # Streamlit interface
├── basic_datamosh.py        # Simple slice-based glitch
├── advanced_datamosh.py     # RGB channel mosh w/ ghosting
├── glitch_effect_stack.py   # Random glitch types
├── frame_overlay_mosher.py  # Overlay-based glitch combo
├── fake_detection_glitch.py # Fake box/label overlay
├── video_mashup_blender.py  # Dual video mashup
├── clip_glitcher_simple.py  # Simple 1s effect clipper
├── requirements.txt
└── .gitignore

💡 Tips

    Press q to stop any live video window if it appears.

    All output videos are downloadable instantly from the browser.

    Dual-video mashups require you to upload two .mp4 files.

🧠 Author

1nc0gn30
🔗 nullai.tech
🐙 github.com/1nc0gn30
🕳️ Future Ideas

Drag-and-drop multiple files

Realtime glitch preview

Upload audio overlays

Export for TikTok/Reels

    Add audio-reactive glitch effects

⚠️ Disclaimer

This app is for creative and educational purposes only. Don’t use it for shady stuff — or do, but don’t tell us 😎.