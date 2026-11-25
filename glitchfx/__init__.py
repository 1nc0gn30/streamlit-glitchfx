"""GlitchFX — deterministic, seed-reproducible image glitch-art engine and CLI."""

from glitchfx.engine import (
    PRESETS,
    PresetSpec,
    apply_pipeline,
    apply_preset,
    bayer_dither,
    crt_scanlines_vignette,
    jpeg_databend,
    list_presets,
    pixel_sort,
    rgb_chromatic_shift,
    scanline_displacement,
)
from glitchfx.samples import SAMPLE_STYLES, generate_sample_image

__all__ = [
    "PRESETS",
    "SAMPLE_STYLES",
    "PresetSpec",
    "apply_pipeline",
    "apply_preset",
    "bayer_dither",
    "crt_scanlines_vignette",
    "generate_sample_image",
    "jpeg_databend",
    "list_presets",
    "pixel_sort",
    "rgb_chromatic_shift",
    "scanline_displacement",
]

__version__ = "2.0.0"
