# GlitchFX (`streamlit-glitchfx`)

Deterministic, seed-reproducible image glitch-art engine, headless CLI, and Streamlit studio built on **Pillow + NumPy**.

Every image transform preserves input dimensions `(W, H)`, keeps RGBA alpha channels intact, and reproduces byte-identical output for a fixed integer seed.

---

## Core Capabilities

- **Pixel-Level Image Glitch Engine (`glitchfx/engine.py`)**:
  - `rgb_chromatic_shift` — Independent R/G/B channel offsets with edge-clamp or wrap mode
  - `scanline_displacement` — Seeded horizontal tape/sync tear bands with optional row jitter
  - `pixel_sort` — Rec.709 luminance-threshold contiguous span sorting (horizontal or vertical, ascending or reverse)
  - `bayer_dither` — Ordered Bayer 4×4 and 8×8 dithering with configurable channel quantization levels
  - `crt_scanlines_vignette` — Horizontal CRT scanlines, RGB aperture-grille phosphor modulation, and radial edge vignette
  - `jpeg_databend` — In-memory JPEG encoding + entropy-stream byte corruption after the SOS marker + 8×8 MCU macroblock scramble
  - `apply_preset` / `apply_pipeline` — Multi-stage deterministic preset and custom pipelines with intensity scaling
- **Procedural Sample Generator (`glitchfx/samples.py`)**: Built-in `neon_grid`, `bauhaus_prism`, and `signal_testcard` artwork so you can test the UI or CLI with zero uploads.
- **Zero-Streamlit Headless CLI (`python -m glitchfx`)**: Single-image, batch directory, and sample-generation workflows with `--preset`, `--seed`, `--intensity`, `--out`, `--format`, and `--list-presets`.
- **Streamlit Studio (`app.py`)**: Live before/after comparison, seed control + randomizer, preset and custom stack controls, lossless PNG + WebP downloads, and an optional Video Datamosh tab when OpenCV/MoviePy are installed.

---

## Installation

### Core Engine + CLI (Pillow + NumPy only)

```bash
pip install -e .
```

### Streamlit Web UI

```bash
pip install -r requirements.txt
```

### Optional Video Datamosh Extras & Dev Tests

```bash
pip install -e ".[app,video,dev]"
```

---

## Streamlit Launch

```bash
streamlit run app.py
```

Open `http://localhost:8501`. The **Image Glitch Lab** opens immediately with a procedural sample canvas so you can test presets, adjust seeds, compare before/after, and export lossless PNG or WebP without uploading anything.

---

## Headless CLI Usage

Run without Streamlit or OpenCV installed:

```bash
# List all built-in presets (add --json for machine-readable output)
python -m glitchfx --list-presets

# Glitch a single image deterministically
python -m glitchfx input.png --preset vhs_tracking --seed 42 --out output.png

# Generate and glitch built-in procedural artwork directly
python -m glitchfx --sample neon_grid --size 1280x720 --preset pixel_melt --seed 7 --out neon_melt.webp

# Batch-process an entire folder into lossless WebP
python -m glitchfx ./photos --preset crt_dither --seed 100 --out ./glitched --format webp
```

If installed via `pip install -e .`, the `glitchfx` console command is also available directly.

---

## Python API

```python
from PIL import Image
from glitchfx import (
    apply_pipeline,
    apply_preset,
    bayer_dither,
    crt_scanlines_vignette,
    generate_sample_image,
    jpeg_databend,
    pixel_sort,
    rgb_chromatic_shift,
    scanline_displacement,
)

# Use your own image or generate a built-in test canvas
img = generate_sample_image("neon_grid", width=800, height=520, seed=42)

# Run a named preset
vhs = apply_preset(img, "vhs_tracking", seed=42, intensity=1.0)
vhs.save("vhs_tracking.png")

# Or compose a custom deterministic pipeline
custom = apply_pipeline(
    img,
    [
        ("pixel_sort", {"low_threshold": 0.25, "high_threshold": 0.82, "direction": "horizontal"}),
        ("bayer_dither", {"matrix_size": 8, "levels": 4, "spread": 1.0}),
        ("rgb_chromatic_shift", {"max_shift": 10, "wrap": False}),
    ],
    seed=1337,
)
custom.save("custom_glitch.webp", format="WEBP", lossless=True)
```

---

## Built-in Presets

| Preset | Label | Pipeline Steps |
| --- | --- | --- |
| `vhs_tracking` | VHS Tracking Tear | `scanline_displacement` → `rgb_chromatic_shift` → `crt_scanlines_vignette` |
| `pixel_melt` | Luminance Pixel Melt | `pixel_sort` (horizontal) → `rgb_chromatic_shift` |
| `databend_raw` | JPEG Databend Raw | `jpeg_databend` → `scanline_displacement` → `rgb_chromatic_shift` |
| `crt_dither` | CRT Bayer 8x8 Dither | `bayer_dither` (8×8) → `crt_scanlines_vignette` |
| `cyber_mosaic` | Vertical Sort + Bayer 4x4 | `pixel_sort` (vertical, reverse) → `bayer_dither` (4×4) → `rgb_chromatic_shift` |
| `signal_collapse` | Full Signal Collapse | `pixel_sort` → `jpeg_databend` → `scanline_displacement` → `rgb_chromatic_shift` → `crt_scanlines_vignette` |

---

## Running Tests

```bash
pytest -v
```

---

## Project Layout

```text
streamlit-glitchfx/
├── app.py                   # Streamlit Image Glitch Lab + optional Video Datamosh tab
├── glitchfx/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # `python -m glitchfx` entry point
│   ├── engine.py            # Deterministic Pillow + NumPy glitch transforms & presets
│   ├── samples.py           # Procedural sample artwork generator
│   └── cli.py               # Zero-Streamlit headless CLI
├── tests/
│   ├── test_engine.py       # Transform, RGBA safety, determinism, and preset tests
│   └── test_cli.py          # End-to-end CLI and app.py syntax tests
├── pyproject.toml           # Package metadata & `glitchfx` console entry point
├── requirements.txt         # Streamlit Cloud / local UI dependencies
└── README.md
```

---

## Attribution & Support

Engineered by Neal Frazier Tech ([@1nc0gn30](https://github.com/1nc0gn30)) • [NullAI Platform](https://nullai.tech)

This project is 100% free and open source. If it helped your creative workflow:
- ☕ [Buy Us a Coffee](https://buy.stripe.com/3cIfZg27PdKo2D64Ocfw40X)
- Solana Tip: `GG2eQbVh8ut1BK6j1A2tz7pSySdLpD38BgyEhGcxdXoK`
