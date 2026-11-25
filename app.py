"""GlitchFX — Deterministic Image Glitch Lab & Optional Video Datamosh Studio."""

from __future__ import annotations

import io
import os
import random
import tempfile
import time
from typing import Any

import streamlit as st
from PIL import Image

from glitchfx import (
    PRESETS,
    SAMPLE_STYLES,
    __version__,
    apply_pipeline,
    apply_preset,
    generate_sample_image,
)


def _encode_image_bytes(image: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "WEBP":
        image.save(buf, format="WEBP", lossless=True, quality=100)
    else:
        image.save(buf, format="PNG")
    return buf.getvalue()


def _load_video_backends() -> tuple[Any, dict[str, Any], dict[str, Any]] | None:
    """Lazily load OpenCV and legacy video datamosh effects when available."""
    try:
        import cv2
        from advanced_datamosh import advanced_datamosh_effect as advanced_effect
        from basic_datamosh import datamosh_effect as basic_effect
        from clip_glitcher_simple import (
            datamosh_effect_1 as clip_1,
            datamosh_effect_2 as clip_2,
        )
        from fake_detection_glitch import (
            datamosh_effect_1 as detection_1,
            datamosh_effect_2 as detection_2,
            fake_detection_overlay,
        )
        from video_mashup_blender import advanced_mash_effect
    except Exception:
        return None

    effects: dict[str, Any] = {
        "Basic Datamosh": basic_effect,
        "Advanced Datamosh": advanced_effect,
        "Clip Glitcher (Moshes)": lambda f: clip_2(clip_1(f)),
        "Fake Detection Glitch": lambda f: fake_detection_overlay(detection_2(detection_1(f))),
    }
    try:
        from frame_overlay_mosher import add_glitch_effect as overlay_effect
        from glitch_effect_stack import add_glitch_effect as glitch_stack_effect

        effects["Glitch Stack"] = glitch_stack_effect
        effects["Frame Overlay Mosher"] = overlay_effect
    except Exception:
        pass

    special: dict[str, Any] = {
        "Video Mashup Blender (2 videos)": advanced_mash_effect,
    }
    return cv2, effects, special


def _apply_video_effect(cv2_mod: Any, input_path: str, output_path: str, effect_func: Any) -> None:
    cap = cv2_mod.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")
    fps = float(cap.get(cv2_mod.CAP_PROP_FPS)) or 24.0
    width = int(cap.get(cv2_mod.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2_mod.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
    out = cv2_mod.VideoWriter(output_path, fourcc, max(1.0, fps), (width, height))
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(effect_func(frame))
    finally:
        cap.release()
        out.release()


def _apply_dual_video_effect(
    cv2_mod: Any, video1_bytes: bytes, video2_bytes: bytes, output_path: str, effect_func: Any
) -> None:
    path1 = path2 = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f1:
            f1.write(video1_bytes)
            path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f2:
            f2.write(video2_bytes)
            path2 = f2.name

        cap1 = cv2_mod.VideoCapture(path1)
        cap2 = cv2_mod.VideoCapture(path2)
        if not cap1.isOpened() or not cap2.isOpened():
            cap1.release()
            cap2.release()
            raise RuntimeError("Failed to open one or both input videos.")

        fps = float(cap1.get(cv2_mod.CAP_PROP_FPS)) or 24.0
        width = int(cap1.get(cv2_mod.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2_mod.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
        out = cv2_mod.VideoWriter(output_path, fourcc, max(1.0, fps), (width, height))
        try:
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break
                frame2 = cv2_mod.resize(frame2, (width, height))
                out.write(effect_func(frame1, frame2))
        finally:
            cap1.release()
            cap2.release()
            out.release()
    finally:
        for tmp_path in (path1, path2):
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def _render_image_lab() -> None:
    if "glitch_seed" not in st.session_state:
        st.session_state.glitch_seed = 42

    with st.sidebar:
        st.header("Source")
        source_mode = st.radio(
            "Image Input",
            ("Procedural Sample", "Upload Image"),
            horizontal=True,
        )
        uploaded_file = None
        sample_style = "neon_grid"
        sample_size = (800, 520)

        if source_mode == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload PNG, JPG, or WebP",
                type=["png", "jpg", "jpeg", "webp"],
            )
        else:
            sample_style = st.selectbox(
                "Sample Artwork",
                options=list(SAMPLE_STYLES.keys()),
                format_func=lambda k: f"{k} — {SAMPLE_STYLES[k][:36]}...",
            )
            res_choice = st.selectbox(
                "Canvas Resolution",
                options=("800x520", "1024x640", "640x640"),
                index=0,
            )
            w_str, h_str = res_choice.split("x")
            sample_size = (int(w_str), int(h_str))

        st.divider()
        st.header("Deterministic Seed")
        seed_col1, seed_col2 = st.columns([2, 1])
        with seed_col2:
            if st.button("Random", use_container_width=True):
                st.session_state.glitch_seed = random.randint(0, 999_999)
        with seed_col1:
            seed = int(
                st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=int(st.session_state.glitch_seed),
                    step=1,
                    label_visibility="collapsed",
                )
            )
            st.session_state.glitch_seed = seed

        st.divider()
        st.header("Pipeline Mode")
        pipeline_mode = st.radio("Mode", ("Preset Pipeline", "Custom Stack"), horizontal=True)

        selected_preset = "vhs_tracking"
        intensity = 1.0
        custom_steps: list[tuple[str, dict[str, Any]]] = []

        if pipeline_mode == "Preset Pipeline":
            preset_keys = list(PRESETS.keys())
            selected_preset = st.selectbox(
                "Preset",
                options=preset_keys,
                format_func=lambda k: f"{PRESETS[k].label} ({k})",
            )
            st.caption(PRESETS[selected_preset].description)
            intensity = st.slider("Intensity", min_value=0.0, max_value=2.0, value=1.0, step=0.05)
        else:
            if st.checkbox("Luminance Pixel Sort", value=True):
                ps_dir = st.selectbox("Sort Direction", ("horizontal", "vertical"))
                ps_low, ps_high = st.slider("Luminance Threshold", 0.0, 1.0, (0.25, 0.80), 0.01)
                ps_rev = st.checkbox("Reverse Sort Order", value=False)
                custom_steps.append(
                    (
                        "pixel_sort",
                        {
                            "low_threshold": ps_low,
                            "high_threshold": ps_high,
                            "direction": ps_dir,
                            "reverse": ps_rev,
                        },
                    )
                )

            if st.checkbox("JPEG Quality / Block Databend", value=False):
                jb_q = st.slider("JPEG Quality", 5, 85, 22)
                jb_c = st.slider("Byte Corruptions", 0, 40, 12)
                jb_b = st.slider("8x8 Macroblock Scramble", 0.0, 0.35, 0.06, 0.01)
                custom_steps.append(
                    (
                        "jpeg_databend",
                        {"quality": jb_q, "corruptions": jb_c, "block_scramble": jb_b},
                    )
                )

            if st.checkbox("Horizontal Scanline Tear", value=True):
                sl_h = st.slider("Band Height (px)", 2, 40, 10)
                sl_s = st.slider("Max Tear Shift (px)", 0, 120, 28)
                sl_p = st.slider("Tear Probability", 0.0, 1.0, 0.35, 0.05)
                custom_steps.append(
                    (
                        "scanline_displacement",
                        {
                            "band_height": sl_h,
                            "max_shift": sl_s,
                            "tear_probability": sl_p,
                            "jitter": 2,
                        },
                    )
                )

            if st.checkbox("RGB Chromatic Shift", value=True):
                cs_max = st.slider("Max Channel Shift (px)", 0, 50, 12)
                cs_wrap = st.checkbox("Wrap Channel Edges", value=False)
                custom_steps.append(
                    ("rgb_chromatic_shift", {"max_shift": cs_max, "wrap": cs_wrap})
                )

            if st.checkbox("Ordered Bayer Dither", value=False):
                bd_sz = st.radio("Bayer Matrix", (4, 8), index=1, horizontal=True)
                bd_lvl = st.slider("Color Levels / Channel", 2, 16, 4)
                bd_sp = st.slider("Dither Spread", 0.0, 2.0, 1.0, 0.1)
                custom_steps.append(
                    ("bayer_dither", {"matrix_size": bd_sz, "levels": bd_lvl, "spread": bd_sp})
                )

            if st.checkbox("CRT Phosphor Scanlines + Vignette", value=True):
                crt_sp = st.slider("Scanline Spacing (px)", 2, 8, 3)
                crt_st = st.slider("Scanline Darkness", 0.0, 0.85, 0.35, 0.05)
                crt_ph = st.slider("Phosphor Mask", 0.0, 0.80, 0.20, 0.05)
                crt_vg = st.slider("Vignette Strength", 0.0, 0.90, 0.45, 0.05)
                custom_steps.append(
                    (
                        "crt_scanlines_vignette",
                        {
                            "scanline_spacing": crt_sp,
                            "scanline_strength": crt_st,
                            "phosphor_strength": crt_ph,
                            "vignette_strength": crt_vg,
                        },
                    )
                )

    # Resolve source image (zero-upload fallback to procedural sample)
    if source_mode == "Upload Image" and uploaded_file is not None:
        source_img = Image.open(uploaded_file)
        source_img.load()
        base_stem = os.path.splitext(uploaded_file.name)[0]
    else:
        if source_mode == "Upload Image":
            st.info("Upload an image in the sidebar, or inspect the built-in procedural sample below.")
        source_img = generate_sample_image(
            style=sample_style,
            width=sample_size[0],
            height=sample_size[1],
            seed=seed,
        )
        base_stem = sample_style

    t0 = time.perf_counter()
    if pipeline_mode == "Preset Pipeline":
        glitched_img = apply_preset(source_img, selected_preset, seed=seed, intensity=intensity)
        tag = selected_preset
    else:
        glitched_img = apply_pipeline(source_img, custom_steps, seed=seed, intensity=1.0)
        tag = "custom"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Telemetry bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dimensions", f"{glitched_img.size[0]} x {glitched_img.size[1]}")
    m2.metric("Color Mode", glitched_img.mode)
    m3.metric("Seed", str(seed))
    m4.metric("Render Time", f"{elapsed_ms:.1f} ms")

    # Live before / after comparison
    view_mode = st.radio(
        "Comparison Layout",
        ("Side-by-Side", "After Only", "Before Only"),
        horizontal=True,
    )
    if view_mode == "Side-by-Side":
        col_before, col_after = st.columns(2)
        with col_before:
            st.subheader("Before")
            st.image(source_img, use_container_width=True)
        with col_after:
            st.subheader("After")
            st.image(glitched_img, use_container_width=True)
    elif view_mode == "After Only":
        st.image(glitched_img, caption=f"Glitched ({tag}, seed={seed})", use_container_width=True)
    else:
        st.image(source_img, caption="Original source", use_container_width=True)

    # Lossless PNG + WebP downloads
    dl_col1, dl_col2 = st.columns(2)
    filename_base = f"{base_stem}_{tag}_s{seed}"
    with dl_col1:
        st.download_button(
            "Download Lossless PNG",
            data=_encode_image_bytes(glitched_img, "PNG"),
            file_name=f"{filename_base}.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "Download Lossless WebP",
            data=_encode_image_bytes(glitched_img, "WEBP"),
            file_name=f"{filename_base}.webp",
            mime="image/webp",
            use_container_width=True,
        )


def _render_video_lab() -> None:
    backend = _load_video_backends()
    if backend is None:
        st.info(
            "Video datamosh requires optional OpenCV (`opencv-python-headless`) and MoviePy dependencies. "
            "Install them with `pip install '.[video]'` to enable MP4 video processing."
        )
        return

    cv2_mod, effects, special_effects = backend
    effect_type = st.selectbox(
        "Choose Video Effect",
        list(effects.keys()) + list(special_effects.keys()),
    )

    if effect_type in effects:
        video_file = st.file_uploader("Upload MP4 video", type=["mp4"], key="single_mp4")
        if video_file and st.button("Apply Video Glitch"):
            with st.spinner("Processing video frames..."):
                in_path = out_path = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                        tmp_in.write(video_file.read())
                        in_path = tmp_in.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                        out_path = tmp_out.name
                    _apply_video_effect(cv2_mod, in_path, out_path, effects[effect_type])
                    with open(out_path, "rb") as out_f:
                        payload = out_f.read()
                    st.success("Video glitch complete.")
                    st.download_button(
                        "Download Glitched MP4",
                        data=payload,
                        file_name=f"{effect_type.lower().replace(' ', '_')}.mp4",
                        mime="video/mp4",
                    )
                except Exception as exc:
                    st.error(f"Video processing failed: {exc}")
                finally:
                    for tmp in (in_path, out_path):
                        if tmp and os.path.exists(tmp):
                            os.unlink(tmp)
    else:
        col1, col2 = st.columns(2)
        with col1:
            vid1 = st.file_uploader("Video 1 (MP4)", type=["mp4"], key="vid1")
        with col2:
            vid2 = st.file_uploader("Video 2 (MP4)", type=["mp4"], key="vid2")
        if vid1 and vid2 and st.button("Mash Videos"):
            with st.spinner("Combining video streams..."):
                out_path = ""
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                        out_path = tmp_out.name
                    _apply_dual_video_effect(
                        cv2_mod,
                        vid1.read(),
                        vid2.read(),
                        out_path,
                        special_effects[effect_type],
                    )
                    with open(out_path, "rb") as out_f:
                        payload = out_f.read()
                    st.success("Video mash complete.")
                    st.download_button(
                        "Download Mashup MP4",
                        data=payload,
                        file_name="mashed_output.mp4",
                        mime="video/mp4",
                    )
                except Exception as exc:
                    st.error(f"Video mash failed: {exc}")
                finally:
                    if out_path and os.path.exists(out_path):
                        os.unlink(out_path)


def main() -> None:
    st.set_page_config(page_title=f"GlitchFX v{__version__}", layout="wide")
    st.title(f"GlitchFX v{__version__}")
    st.caption(
        "Deterministic pixel-level image glitch engine (Pillow + NumPy) with built-in sample art, "
        "seed reproducibility, lossless PNG/WebP export, and optional MP4 datamosh."
    )

    tab_image, tab_video = st.tabs(["Image Glitch Lab", "Video Datamosh Lab (Optional)"])
    with tab_image:
        _render_image_lab()
    with tab_video:
        _render_video_lab()

    st.divider()
    st.caption(
        "Engineered by Neal Frazier Tech (@1nc0gn30) • NullAI Platform | "
        "[GitHub](https://github.com/1nc0gn30/streamlit-glitchfx) • "
        "[nullai.tech](https://nullai.tech) • "
        "[☕ Buy Us a Coffee](https://buy.stripe.com/3cIfZg27PdKo2D64Ocfw40X) • "
        "SOL: `GG2eQbVh8ut1BK6j1A2tz7pSySdLpD38BgyEhGcxdXoK`"
    )


if __name__ == "__main__":
    main()
