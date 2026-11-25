"""Procedural sample artwork generator (Pillow + NumPy, zero external assets)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

SAMPLE_STYLES: dict[str, str] = {
    "neon_grid": "Synthwave horizon grid, gradient sky, sun orb, and geometric monoliths",
    "bauhaus_prism": "High-contrast geometric shapes, diagonals, and luminance gradients",
    "signal_testcard": "Broadcast color bars, circular target, grayscale ramp, and crosshair grid",
}


def _generate_neon_grid(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    horizon = 0.56

    img = np.zeros((height, width, 3), dtype=np.float32)

    # Sky gradient: deep indigo -> magenta -> warm sunset orange near horizon
    sky_t = np.clip(y / horizon, 0.0, 1.0)
    sky_r = 14.0 + 185.0 * (sky_t**2.1) + 40.0 * np.sin(sky_t * np.pi)
    sky_g = 12.0 + 52.0 * (sky_t**2.4)
    sky_b = 46.0 + 95.0 * (1.0 - sky_t) + 75.0 * np.sin(sky_t * np.pi)

    # Ground gradient: dark violet fading to black
    gnd_t = np.clip((y - horizon) / max(1e-5, 1.0 - horizon), 0.0, 1.0)
    gnd_r = 26.0 * (1.0 - gnd_t * 0.7)
    gnd_g = 8.0 * (1.0 - gnd_t * 0.5)
    gnd_b = 48.0 * (1.0 - gnd_t * 0.65)

    is_sky = y < horizon
    img[:, :, 0] = np.where(is_sky, sky_r, gnd_r)
    img[:, :, 1] = np.where(is_sky, sky_g, gnd_g)
    img[:, :, 2] = np.where(is_sky, sky_b, gnd_b)

    # Deterministic starfield in upper sky
    num_stars = max(25, (width * height) // 4500)
    star_y = rng.integers(0, max(1, int(height * horizon * 0.72)), size=num_stars)
    star_x = rng.integers(0, width, size=num_stars)
    star_b = rng.uniform(160.0, 255.0, size=num_stars)
    img[star_y, star_x, 0] = star_b
    img[star_y, star_x, 1] = star_b * 0.96
    img[star_y, star_x, 2] = 255.0

    # Retro sun orb with horizontal cutout bands
    cx, cy = 0.50, horizon - 0.14
    aspect = width / max(1.0, float(height))
    dx = (x - cx) * aspect
    dy = y - cy
    dist = np.sqrt(dx * dx + dy * dy)
    radius = 0.21
    in_sun = (dist <= radius) & is_sky
    sun_t = np.clip((y - (cy - radius)) / (2.0 * radius), 0.0, 1.0)
    # Horizontal sun scan cutouts on lower half
    stripe_mask = ~((sun_t > 0.45) & (np.sin(sun_t * 52.0) > 0.55))
    sun_mask = in_sun & stripe_mask
    img[:, :, 0] = np.where(sun_mask, 255.0, img[:, :, 0])
    img[:, :, 1] = np.where(sun_mask, 235.0 * (1.0 - 0.72 * sun_t), img[:, :, 1])
    img[:, :, 2] = np.where(sun_mask, 90.0 * (1.0 - sun_t) + 85.0 * sun_t, img[:, :, 2])

    # Horizon glow
    glow = np.exp(-((y - horizon) ** 2) / 0.0012)
    img[:, :, 0] += 130.0 * glow
    img[:, :, 1] += 40.0 * glow
    img[:, :, 2] += 110.0 * glow

    # Perspective ground grid
    depth = np.maximum(0.015, y - horizon)
    u = (x - 0.5) / depth
    v = 0.18 / depth
    grid_u = np.abs(np.sin(u * np.pi * 1.4)) < 0.055
    grid_v = np.abs(np.sin(v * np.pi * 1.1)) < 0.07
    grid = (~is_sky) & (grid_u | grid_v)
    fade = np.clip(gnd_t * 2.2, 0.2, 1.0)
    img[:, :, 0] = np.where(grid, 60.0 + 180.0 * (1.0 - fade), img[:, :, 0])
    img[:, :, 1] = np.where(grid, 220.0 * fade + 40.0, img[:, :, 1])
    img[:, :, 2] = np.where(grid, 255.0, img[:, :, 2])

    pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    # Geometric foreground silhouettes / monoliths for strong pixel-sort edges
    hy = int(height * horizon)
    draw.polygon(
        [(0, hy), (int(width * 0.14), int(hy * 0.72)), (int(width * 0.30), hy)],
        fill=(22, 12, 42),
        outline=(255, 70, 175),
    )
    draw.polygon(
        [(int(width * 0.68), hy), (int(width * 0.85), int(hy * 0.68)), (width, hy)],
        fill=(18, 14, 46),
        outline=(55, 235, 255),
    )
    return pil_img


def _generate_bauhaus_prism(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Warm cream/ivory diagonal textured backdrop
    bg = np.zeros((height, width, 3), dtype=np.float32)
    diag = 0.5 * (x + y)
    bg[:, :, 0] = 236.0 - 24.0 * diag
    bg[:, :, 1] = 228.0 - 30.0 * diag
    bg[:, :, 2] = 212.0 - 28.0 * diag
    grain = rng.normal(0.0, 3.0, size=(height, width, 1))
    bg = np.clip(bg + grain, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    # Bold Bauhaus palette blocks
    draw.rectangle([0, 0, int(width * 0.28), height], fill=(22, 26, 36))
    draw.rectangle(
        [int(width * 0.28), int(height * 0.66), width, height],
        fill=(214, 52, 46),
    )
    draw.rectangle(
        [int(width * 0.68), 0, width, int(height * 0.66)],
        fill=(28, 92, 178),
    )

    # Large intersecting circle + ring
    cx, cy = int(width * 0.50), int(height * 0.44)
    r = int(min(width, height) * 0.29)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(242, 188, 38))
    r2 = int(r * 0.52)
    draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=(20, 22, 28))

    # Diagonal prism bar + vertical luminance stripes
    draw.polygon(
        [
            (int(width * 0.12), int(height * 0.84)),
            (int(width * 0.82), int(height * 0.14)),
            (int(width * 0.89), int(height * 0.22)),
            (int(width * 0.19), int(height * 0.92)),
        ],
        fill=(245, 244, 240),
    )
    bar_w = max(2, int(width * 0.02))
    for i in range(8):
        tone = int(30 + i * 31)
        x0 = int(width * 0.05) + i * bar_w
        draw.rectangle([x0, int(height * 0.12), x0 + bar_w, int(height * 0.54)], fill=(tone, tone, tone))

    return pil_img


def _generate_signal_testcard(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    del rng  # Fully deterministic broadcast geometry
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Top 68%: SMPTE-style color bars
    bar_colors = [
        (235, 235, 235),  # White
        (235, 235, 16),   # Yellow
        (16, 235, 235),   # Cyan
        (16, 235, 16),    # Green
        (235, 16, 235),   # Magenta
        (235, 16, 16),    # Red
        (16, 16, 235),    # Blue
    ]
    top_h = int(height * 0.68)
    n_bars = len(bar_colors)
    for idx, color in enumerate(bar_colors):
        x0 = (idx * width) // n_bars
        x1 = ((idx + 1) * width) // n_bars
        img[:top_h, x0:x1] = color

    # Middle 10%: reverse chromatic strip
    mid_h = int(height * 0.78)
    rev_colors = [
        (16, 16, 235),
        (19, 19, 19),
        (235, 16, 235),
        (19, 19, 19),
        (16, 235, 235),
        (19, 19, 19),
        (235, 235, 235),
    ]
    for idx, color in enumerate(rev_colors):
        x0 = (idx * width) // n_bars
        x1 = ((idx + 1) * width) // n_bars
        img[top_h:mid_h, x0:x1] = color

    # Bottom 22%: continuous grayscale luminance ramp
    ramp = np.linspace(0, 255, width, dtype=np.uint8)
    img[mid_h:, :, 0] = ramp[None, :]
    img[mid_h:, :, 1] = ramp[None, :]
    img[mid_h:, :, 2] = ramp[None, :]

    pil_img = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil_img)

    # Center resolution target + crosshair
    cx, cy = width // 2, height // 2
    r_outer = max(12, int(min(width, height) * 0.24))
    draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        fill=(24, 24, 28),
        outline=(245, 245, 245),
        width=max(2, min(width, height) // 160),
    )
    for scale in (0.72, 0.44, 0.18):
        rr = max(2, int(r_outer * scale))
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(230, 230, 230), width=2)

    draw.line([(cx - r_outer, cy), (cx + r_outer, cy)], fill=(255, 255, 255), width=2)
    draw.line([(cx, cy - r_outer), (cx, cy + r_outer)], fill=(255, 255, 255), width=2)

    return pil_img


def generate_sample_image(
    style: str = "neon_grid",
    width: int = 800,
    height: int = 520,
    seed: int = 42,
) -> Image.Image:
    """Generate a deterministic procedural RGB sample image for instant testing."""
    if width < 16 or height < 16:
        raise ValueError(f"Sample image dimensions must be at least 16x16, got {width}x{height}.")
    if style not in SAMPLE_STYLES:
        valid = ", ".join(sorted(SAMPLE_STYLES))
        raise ValueError(f"Unknown sample style '{style}'. Choose one of: {valid}.")

    rng = np.random.default_rng(seed)
    if style == "neon_grid":
        return _generate_neon_grid(width, height, rng)
    if style == "bauhaus_prism":
        return _generate_bauhaus_prism(width, height, rng)
    return _generate_signal_testcard(width, height, rng)
