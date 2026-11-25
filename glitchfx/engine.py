"""Deterministic, seed-reproducible image glitch-art transforms (Pillow + NumPy)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFile


@dataclass(frozen=True)
class PresetSpec:
    """Metadata and transform pipeline definition for a named glitch preset."""

    name: str
    label: str
    description: str
    steps: tuple[tuple[str, dict[str, Any]], ...]


def _split_rgb_alpha(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract an (H, W, 3) uint8 RGB array and optional (H, W) uint8 alpha channel."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}.")
    if image.mode in ("RGBA", "LA"):
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8).copy()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return rgb, alpha
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return rgb, None


def _merge_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> Image.Image:
    """Reassemble a PIL Image in RGB or RGBA mode matching the original alpha presence."""
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    if alpha is not None:
        rgba = np.dstack((rgb_u8, alpha.astype(np.uint8)))
        return Image.fromarray(rgba, mode="RGBA")
    return Image.fromarray(rgb_u8, mode="RGB")


def _shift_channel(channel: np.ndarray, dx: int, dy: int, wrap: bool = False) -> np.ndarray:
    """Shift a 2D channel by (dx, dy) pixels with wrap-around or edge clamping."""
    if dx == 0 and dy == 0:
        return channel.copy()
    if wrap:
        return np.roll(np.roll(channel, dy, axis=0), dx, axis=1)
    h, w = channel.shape
    ys = np.clip(np.arange(h, dtype=np.int64) - int(dy), 0, h - 1)
    xs = np.clip(np.arange(w, dtype=np.int64) - int(dx), 0, w - 1)
    return channel[np.ix_(ys, xs)]


def rgb_chromatic_shift(
    image: Image.Image,
    *,
    red_offset: tuple[int, int] | None = None,
    green_offset: tuple[int, int] | None = None,
    blue_offset: tuple[int, int] | None = None,
    max_shift: int = 12,
    wrap: bool = False,
    seed: int = 42,
) -> Image.Image:
    """Apply RGB chromatic channel displacement with deterministic seeded defaults."""
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0.")

    rgb, alpha = _split_rgb_alpha(image)
    rng = np.random.default_rng(seed)

    if red_offset is None and green_offset is None and blue_offset is None and max_shift == 0:
        return _merge_rgb_alpha(rgb, alpha)

    if red_offset is None or blue_offset is None:
        if max_shift == 0:
            rx = ry = bx = by = 0
        else:
            mag_x = int(rng.integers(max(1, max_shift // 2), max_shift + 1))
            sign = int(rng.choice([-1, 1]))
            rx = mag_x * sign
            y_bound = max(1, max_shift // 3)
            ry = int(rng.integers(-y_bound, y_bound + 1))
            bx, by = -rx, -ry
    else:
        rx, ry = red_offset
        bx, by = blue_offset

    if red_offset is not None:
        rx, ry = red_offset
    if blue_offset is not None:
        bx, by = blue_offset
    if green_offset is not None:
        gx, gy = green_offset
    elif max_shift > 0:
        g_bound = max(1, max_shift // 5)
        gx = int(rng.integers(-g_bound, g_bound + 1))
        gy = 0
    else:
        gx = gy = 0

    out = np.empty_like(rgb)
    out[:, :, 0] = _shift_channel(rgb[:, :, 0], rx, ry, wrap=wrap)
    out[:, :, 1] = _shift_channel(rgb[:, :, 1], gx, gy, wrap=wrap)
    out[:, :, 2] = _shift_channel(rgb[:, :, 2], bx, by, wrap=wrap)
    return _merge_rgb_alpha(out, alpha)


def scanline_displacement(
    image: Image.Image,
    *,
    band_height: int = 10,
    max_shift: int = 28,
    tear_probability: float = 0.35,
    jitter: int = 2,
    seed: int = 42,
) -> Image.Image:
    """Displace horizontal scanline bands to create analog VHS/sync tear artifacts."""
    if band_height < 1:
        raise ValueError("band_height must be >= 1.")
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0.")
    if not (0.0 <= tear_probability <= 1.0):
        raise ValueError("tear_probability must be in [0.0, 1.0].")
    if jitter < 0:
        raise ValueError("jitter must be >= 0.")

    rgb, alpha = _split_rgb_alpha(image)
    h, w = rgb.shape[:2]
    if max_shift == 0 or tear_probability == 0.0 or w < 2:
        return _merge_rgb_alpha(rgb, alpha)

    rng = np.random.default_rng(seed)
    out = rgb.copy()
    y = 0
    min_h = max(1, band_height // 2)
    max_h = max(min_h + 1, band_height * 2 + 1)

    while y < h:
        slice_h = int(rng.integers(min_h, max_h))
        y_end = min(h, y + slice_h)
        if float(rng.random()) < tear_probability:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            if shift == 0:
                shift = int(rng.choice([-1, 1])) * max(1, max_shift // 2)
            out[y:y_end] = np.roll(out[y:y_end], shift, axis=1)
            if jitter > 0:
                for row in range(y, y_end):
                    row_jit = int(rng.integers(-jitter, jitter + 1))
                    if row_jit != 0:
                        out[row] = np.roll(out[row], row_jit, axis=0)
        y = y_end

    return _merge_rgb_alpha(out, alpha)


def _compute_sort_key(rgb: np.ndarray, sort_by: str) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    if sort_by == "luminance":
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    if sort_by == "red":
        return r / 255.0
    if sort_by == "green":
        return g / 255.0
    if sort_by == "blue":
        return b / 255.0
    if sort_by == "saturation":
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        return np.where(cmax > 0.0, (cmax - cmin) / np.maximum(cmax, 1.0), 0.0)
    raise ValueError(
        f"Unsupported sort_by '{sort_by}'. Choose from: luminance, red, green, blue, saturation."
    )


def pixel_sort(
    image: Image.Image,
    *,
    low_threshold: float = 0.25,
    high_threshold: float = 0.80,
    direction: str = "horizontal",
    reverse: bool = False,
    sort_by: str = "luminance",
) -> Image.Image:
    """Sort contiguous pixel spans whose Rec.709 luminance falls within [low, high]."""
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be 'horizontal' or 'vertical'.")
    if not (0.0 <= low_threshold <= high_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy 0.0 <= low_threshold <= high_threshold <= 1.0.")

    rgb, alpha = _split_rgb_alpha(image)
    lum = (
        0.2126 * rgb[:, :, 0].astype(np.float32)
        + 0.7152 * rgb[:, :, 1].astype(np.float32)
        + 0.0722 * rgb[:, :, 2].astype(np.float32)
    ) / 255.0
    keys = _compute_sort_key(rgb, sort_by)
    mask = (lum >= low_threshold) & (lum <= high_threshold)

    work = rgb.copy()
    if direction == "vertical":
        work = np.transpose(work, (1, 0, 2)).copy()
        keys = keys.T
        mask = mask.T

    n_rows = work.shape[0]
    for row in range(n_rows):
        row_mask = mask[row]
        if not np.any(row_mask):
            continue
        padded = np.empty(row_mask.size + 2, dtype=np.bool_)
        padded[0] = False
        padded[1:-1] = row_mask
        padded[-1] = False
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts = edges[0::2]
        ends = edges[1::2]
        row_keys = keys[row]
        row_pixels = work[row]
        for start, end in zip(starts, ends):
            if end - start <= 1:
                continue
            order = np.argsort(row_keys[start:end], kind="mergesort")
            if reverse:
                order = order[::-1]
            row_pixels[start:end] = row_pixels[start:end][order]

    if direction == "vertical":
        work = np.transpose(work, (1, 0, 2)).copy()

    return _merge_rgb_alpha(work, alpha)


def _build_bayer_matrix(size: int) -> np.ndarray:
    """Recursively build an ordered Bayer threshold matrix of size 2^k."""
    if size == 2:
        return np.array([[0, 2], [3, 1]], dtype=np.float32)
    half = _build_bayer_matrix(size // 2)
    return np.block(
        [
            [4.0 * half + 0.0, 4.0 * half + 2.0],
            [4.0 * half + 3.0, 4.0 * half + 1.0],
        ]
    )


def bayer_dither(
    image: Image.Image,
    *,
    matrix_size: int = 8,
    levels: int = 4,
    spread: float = 1.0,
) -> Image.Image:
    """Apply ordered Bayer 4x4 or 8x8 dithering with per-channel quantization."""
    if matrix_size not in (4, 8):
        raise ValueError("matrix_size must be 4 or 8.")
    if levels < 2 or levels > 256:
        raise ValueError("levels must be between 2 and 256.")
    if spread < 0.0:
        raise ValueError("spread must be >= 0.0.")

    rgb, alpha = _split_rgb_alpha(image)
    h, w = rgb.shape[:2]

    bayer = _build_bayer_matrix(matrix_size)
    threshold = (bayer + 0.5) / float(matrix_size * matrix_size) - 0.5
    tiled = np.tile(threshold, (h // matrix_size + 1, w // matrix_size + 1))[:h, :w]

    norm = rgb.astype(np.float32) / 255.0
    step = 1.0 / float(levels - 1)
    dithered = np.clip(norm + tiled[:, :, None] * spread * step, 0.0, 1.0)
    quantized = np.round(dithered * (levels - 1)) * step
    out = np.clip(np.round(quantized * 255.0), 0, 255).astype(np.uint8)
    return _merge_rgb_alpha(out, alpha)


def crt_scanlines_vignette(
    image: Image.Image,
    *,
    scanline_spacing: int = 3,
    scanline_strength: float = 0.35,
    phosphor_strength: float = 0.20,
    vignette_strength: float = 0.45,
) -> Image.Image:
    """Simulate CRT horizontal scanlines, RGB phosphor aperture grille, and radial vignette."""
    if scanline_spacing < 2:
        raise ValueError("scanline_spacing must be >= 2.")
    if not (0.0 <= scanline_strength <= 1.0):
        raise ValueError("scanline_strength must be in [0.0, 1.0].")
    if not (0.0 <= phosphor_strength <= 1.0):
        raise ValueError("phosphor_strength must be in [0.0, 1.0].")
    if not (0.0 <= vignette_strength <= 1.0):
        raise ValueError("vignette_strength must be in [0.0, 1.0].")

    rgb, alpha = _split_rgb_alpha(image)
    h, w = rgb.shape[:2]
    work = rgb.astype(np.float32) / 255.0

    # Horizontal CRT scanlines
    scanline_mask = np.ones((h, 1, 1), dtype=np.float32)
    scanline_mask[::scanline_spacing, :, :] = 1.0 - float(scanline_strength)
    work *= scanline_mask

    # RGB aperture-grille phosphor subpixel modulation
    if phosphor_strength > 0.0:
        cols = np.arange(w) % 3
        phosphor = np.full((1, w, 3), 1.0 - 0.5 * phosphor_strength, dtype=np.float32)
        phosphor[0, cols == 0, 0] = 1.0 + 0.35 * phosphor_strength
        phosphor[0, cols == 1, 1] = 1.0 + 0.35 * phosphor_strength
        phosphor[0, cols == 2, 2] = 1.0 + 0.35 * phosphor_strength
        work *= phosphor

    # Radial edge vignette
    if vignette_strength > 0.0:
        nx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        ny = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        r2 = (nx[None, :] ** 2 + ny[:, None] ** 2) * 0.5
        vignette = np.clip(1.0 - float(vignette_strength) * r2, 0.0, 1.0)
        work *= vignette[:, :, None]

    out = np.clip(np.round(work * 255.0), 0, 255).astype(np.uint8)
    return _merge_rgb_alpha(out, alpha)


def _find_jpeg_scan_bounds(data: bytearray) -> tuple[int, int] | None:
    """Return (scan_data_start, eoi_index) inside a JPEG byte buffer, or None."""
    n = len(data)
    if n < 32 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    pos = 2
    scan_start: int | None = None
    while pos + 3 < n:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xD9:  # EOI
            break
        if marker in (0x00, 0x01) or (0xD0 <= marker <= 0xD7):
            pos += 2
            continue
        seg_len = (data[pos + 2] << 8) | data[pos + 3]
        if marker == 0xDA:  # Start of Scan (SOS)
            scan_start = pos + 2 + seg_len
            break
        pos += 2 + seg_len

    if scan_start is None:
        return None

    eoi = data.rfind(b"\xff\xd9")
    if eoi == -1 or eoi <= scan_start:
        return None
    return scan_start, eoi


def jpeg_databend(
    image: Image.Image,
    *,
    quality: int = 24,
    corruptions: int = 10,
    block_scramble: float = 0.06,
    seed: int = 42,
) -> Image.Image:
    """Apply real in-memory JPEG compression + entropy-stream databend + 8x8 macroblock shift."""
    if not (1 <= quality <= 95):
        raise ValueError("quality must be between 1 and 95.")
    if corruptions < 0:
        raise ValueError("corruptions must be >= 0.")
    if not (0.0 <= block_scramble <= 1.0):
        raise ValueError("block_scramble must be in [0.0, 1.0].")

    rgb, alpha = _split_rgb_alpha(image)
    h, w = rgb.shape[:2]
    rng = np.random.default_rng(seed)

    base_pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    base_pil.save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0")
    raw_bytes = buf.getvalue()

    corrupted = bytearray(raw_bytes)
    bounds = _find_jpeg_scan_bounds(corrupted)
    if bounds is not None and corruptions > 0:
        scan_start, eoi = bounds
        safe_start = scan_start + 12
        safe_end = eoi - 12
        if safe_end > safe_start:
            # Candidate byte positions that do not touch 0xFF marker escapes
            applied = 0
            attempts = 0
            max_attempts = corruptions * 12
            while applied < corruptions and attempts < max_attempts:
                attempts += 1
                idx = int(rng.integers(safe_start, safe_end))
                if corrupted[idx] == 0xFF or corrupted[idx - 1] == 0xFF:
                    continue
                new_val = int(rng.integers(1, 254))
                if new_val == corrupted[idx]:
                    new_val = (new_val % 253) + 1
                corrupted[idx] = new_val
                applied += 1

    prev_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            decoded = Image.open(io.BytesIO(bytes(corrupted)))
            decoded.load()
            decoded_rgb = decoded.convert("RGB")
        except OSError:
            decoded = Image.open(io.BytesIO(raw_bytes))
            decoded.load()
            decoded_rgb = decoded.convert("RGB")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev_truncated

    if decoded_rgb.size != (w, h):
        decoded_rgb = decoded_rgb.resize((w, h), Image.Resampling.NEAREST)

    out = np.asarray(decoded_rgb, dtype=np.uint8).copy()

    # Optional 8x8 macroblock permutation aligned to JPEG MCU grid
    if block_scramble > 0.0 and h >= 8 and w >= 8:
        bh = h // 8
        bw = w // 8
        total_blocks = bh * bw
        n_swaps = max(1, int(total_blocks * block_scramble)) if total_blocks > 1 else 0
        for _ in range(n_swaps):
            by = int(rng.integers(0, bh))
            bx = int(rng.integers(0, bw))
            dy = int(rng.integers(-2, 3))
            dx = int(rng.integers(-4, 5))
            ty = int(np.clip(by + dy, 0, bh - 1))
            tx = int(np.clip(bx + dx, 0, bw - 1))
            if (by, bx) == (ty, tx):
                continue
            y0, y1 = by * 8, (by + 1) * 8
            x0, x1 = bx * 8, (bx + 1) * 8
            ty0, ty1 = ty * 8, (ty + 1) * 8
            tx0, tx1 = tx * 8, (tx + 1) * 8
            tmp = out[y0:y1, x0:x1].copy()
            out[y0:y1, x0:x1] = out[ty0:ty1, tx0:tx1]
            out[ty0:ty1, tx0:tx1] = tmp

    return _merge_rgb_alpha(out, alpha)


_TRANSFORMS: dict[str, Callable[..., Image.Image]] = {
    "rgb_chromatic_shift": rgb_chromatic_shift,
    "scanline_displacement": scanline_displacement,
    "pixel_sort": pixel_sort,
    "bayer_dither": bayer_dither,
    "crt_scanlines_vignette": crt_scanlines_vignette,
    "jpeg_databend": jpeg_databend,
}

_SEEDED_TRANSFORMS = frozenset(
    {"rgb_chromatic_shift", "scanline_displacement", "jpeg_databend"}
)

PRESETS: dict[str, PresetSpec] = {
    "vhs_tracking": PresetSpec(
        name="vhs_tracking",
        label="VHS Tracking Tear",
        description="Analog horizontal tape tear + RGB chromatic shift + CRT phosphor scanlines",
        steps=(
            ("scanline_displacement", {"band_height": 10, "max_shift": 26, "tear_probability": 0.35, "jitter": 2}),
            ("rgb_chromatic_shift", {"max_shift": 10, "wrap": False}),
            ("crt_scanlines_vignette", {"scanline_spacing": 3, "scanline_strength": 0.34, "phosphor_strength": 0.22, "vignette_strength": 0.42}),
        ),
    ),
    "pixel_melt": PresetSpec(
        name="pixel_melt",
        label="Luminance Pixel Melt",
        description="Horizontal Rec.709 luminance pixel sorting paired with subtle chromatic split",
        steps=(
            ("pixel_sort", {"low_threshold": 0.24, "high_threshold": 0.82, "direction": "horizontal", "reverse": False}),
            ("rgb_chromatic_shift", {"max_shift": 7, "wrap": False}),
        ),
    ),
    "databend_raw": PresetSpec(
        name="databend_raw",
        label="JPEG Databend Raw",
        description="Low-quality JPEG entropy stream corruption + 8x8 macroblock scramble + sync tear",
        steps=(
            ("jpeg_databend", {"quality": 18, "corruptions": 14, "block_scramble": 0.07}),
            ("scanline_displacement", {"band_height": 12, "max_shift": 22, "tear_probability": 0.28, "jitter": 1}),
            ("rgb_chromatic_shift", {"max_shift": 9, "wrap": False}),
        ),
    ),
    "crt_dither": PresetSpec(
        name="crt_dither",
        label="CRT Bayer 8x8 Dither",
        description="Ordered Bayer 8x8 quantization into crisp retro palette + CRT phosphor mask",
        steps=(
            ("bayer_dither", {"matrix_size": 8, "levels": 4, "spread": 1.0}),
            ("crt_scanlines_vignette", {"scanline_spacing": 3, "scanline_strength": 0.38, "phosphor_strength": 0.28, "vignette_strength": 0.48}),
        ),
    ),
    "cyber_mosaic": PresetSpec(
        name="cyber_mosaic",
        label="Vertical Sort + Bayer 4x4",
        description="Vertical luminance pixel sorting + ordered Bayer 4x4 dither + RGB split",
        steps=(
            ("pixel_sort", {"low_threshold": 0.28, "high_threshold": 0.85, "direction": "vertical", "reverse": True}),
            ("bayer_dither", {"matrix_size": 4, "levels": 5, "spread": 0.9}),
            ("rgb_chromatic_shift", {"max_shift": 11, "wrap": False}),
        ),
    ),
    "signal_collapse": PresetSpec(
        name="signal_collapse",
        label="Full Signal Collapse",
        description="Complete glitch stack: pixel sort + JPEG databend + scanline tear + RGB shift + CRT",
        steps=(
            ("pixel_sort", {"low_threshold": 0.32, "high_threshold": 0.84, "direction": "horizontal"}),
            ("jpeg_databend", {"quality": 16, "corruptions": 16, "block_scramble": 0.08}),
            ("scanline_displacement", {"band_height": 8, "max_shift": 34, "tear_probability": 0.42, "jitter": 3}),
            ("rgb_chromatic_shift", {"max_shift": 14, "wrap": False}),
            ("crt_scanlines_vignette", {"scanline_spacing": 3, "scanline_strength": 0.35, "phosphor_strength": 0.22, "vignette_strength": 0.46}),
        ),
    ),
}


def list_presets() -> list[PresetSpec]:
    """Return all registered preset specifications in display order."""
    return list(PRESETS.values())


def _scale_step_params(name: str, params: Mapping[str, Any], intensity: float) -> dict[str, Any]:
    """Scale numeric effect strength parameters cleanly according to intensity."""
    scaled = dict(params)
    if intensity == 1.0:
        return scaled

    if name == "rgb_chromatic_shift" and "max_shift" in scaled:
        scaled["max_shift"] = max(0, int(round(scaled["max_shift"] * intensity)))
    elif name == "scanline_displacement":
        if "max_shift" in scaled:
            scaled["max_shift"] = max(0, int(round(scaled["max_shift"] * intensity)))
        if "tear_probability" in scaled:
            scaled["tear_probability"] = float(np.clip(scaled["tear_probability"] * intensity, 0.0, 1.0))
    elif name == "jpeg_databend":
        if "corruptions" in scaled:
            scaled["corruptions"] = max(0, int(round(scaled["corruptions"] * intensity)))
        if "block_scramble" in scaled:
            scaled["block_scramble"] = float(np.clip(scaled["block_scramble"] * intensity, 0.0, 1.0))
    elif name == "crt_scanlines_vignette":
        for key in ("scanline_strength", "phosphor_strength", "vignette_strength"):
            if key in scaled:
                scaled[key] = float(np.clip(scaled[key] * intensity, 0.0, 1.0))
    elif name == "bayer_dither" and "spread" in scaled:
        scaled["spread"] = max(0.0, float(scaled["spread"] * intensity))
    return scaled


def apply_pipeline(
    image: Image.Image,
    steps: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    seed: int = 42,
    intensity: float = 1.0,
) -> Image.Image:
    """Run an ordered sequence of named glitch transforms deterministically."""
    if intensity < 0.0:
        raise ValueError("intensity must be >= 0.0.")

    current = image.copy()
    if intensity == 0.0 or not steps:
        return current

    seed_rng = np.random.default_rng(seed)
    for step_name, raw_params in steps:
        if step_name not in _TRANSFORMS:
            valid = ", ".join(sorted(_TRANSFORMS))
            raise ValueError(f"Unknown transform '{step_name}'. Valid transforms: {valid}.")
        params = _scale_step_params(step_name, raw_params, intensity)
        if step_name in _SEEDED_TRANSFORMS and "seed" not in params:
            params["seed"] = int(seed_rng.integers(0, 2**31 - 1))
        current = _TRANSFORMS[step_name](current, **params)
    return current


def apply_preset(
    image: Image.Image,
    preset: str,
    *,
    seed: int = 42,
    intensity: float = 1.0,
) -> Image.Image:
    """Apply a named glitch preset pipeline to a PIL image."""
    if preset not in PRESETS:
        valid = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {valid}.")
    spec = PRESETS[preset]
    return apply_pipeline(image, spec.steps, seed=seed, intensity=intensity)
