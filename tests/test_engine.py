"""Unit and property tests for glitchfx.engine and glitchfx.samples."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from glitchfx.engine import (
    PRESETS,
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


@pytest.fixture()
def rgb_image() -> Image.Image:
    return generate_sample_image("neon_grid", width=160, height=120, seed=7)


@pytest.fixture()
def rgba_image(rgb_image: Image.Image) -> Image.Image:
    h, w = rgb_image.size[1], rgb_image.size[0]
    alpha = np.linspace(0, 255, w, dtype=np.uint8)[None, :].repeat(h, axis=0)
    rgb_arr = np.asarray(rgb_image, dtype=np.uint8)
    return Image.fromarray(np.dstack((rgb_arr, alpha)), mode="RGBA")


def _assert_same_image(a: Image.Image, b: Image.Image) -> None:
    assert a.size == b.size
    assert a.mode == b.mode
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _assert_alpha_preserved(src: Image.Image, dst: Image.Image) -> None:
    assert dst.mode == "RGBA"
    assert dst.size == src.size
    np.testing.assert_array_equal(
        np.asarray(src.getchannel("A")),
        np.asarray(dst.getchannel("A")),
    )


class TestSampleGenerator:
    @pytest.mark.parametrize("style", sorted(SAMPLE_STYLES.keys()))
    def test_styles_and_determinism(self, style: str) -> None:
        img1 = generate_sample_image(style, width=128, height=96, seed=99)
        img2 = generate_sample_image(style, width=128, height=96, seed=99)
        assert img1.size == (128, 96)
        assert img1.mode == "RGB"
        _assert_same_image(img1, img2)

    def test_invalid_style_and_size(self) -> None:
        with pytest.raises(ValueError):
            generate_sample_image("nonexistent")
        with pytest.raises(ValueError):
            generate_sample_image("neon_grid", width=8, height=8)


class TestRgbChromaticShift:
    def test_determinism_and_rgb_safety(self, rgb_image: Image.Image) -> None:
        a = rgb_chromatic_shift(rgb_image, max_shift=10, seed=123)
        b = rgb_chromatic_shift(rgb_image, max_shift=10, seed=123)
        c = rgb_chromatic_shift(rgb_image, max_shift=10, seed=456)
        _assert_same_image(a, b)
        assert not np.array_equal(np.asarray(a), np.asarray(rgb_image))
        assert not np.array_equal(np.asarray(a), np.asarray(c))

    def test_explicit_offsets_and_wrap(self, rgb_image: Image.Image) -> None:
        shifted_clamp = rgb_chromatic_shift(
            rgb_image,
            red_offset=(8, 0),
            green_offset=(0, 0),
            blue_offset=(-8, 0),
            wrap=False,
        )
        shifted_wrap = rgb_chromatic_shift(
            rgb_image,
            red_offset=(8, 0),
            green_offset=(0, 0),
            blue_offset=(-8, 0),
            wrap=True,
        )
        assert shifted_clamp.size == rgb_image.size
        assert shifted_wrap.size == rgb_image.size
        np.testing.assert_array_equal(
            np.asarray(shifted_clamp)[:, :, 1],
            np.asarray(rgb_image)[:, :, 1],
        )

    def test_zero_shift_identity(self, rgb_image: Image.Image) -> None:
        same = rgb_chromatic_shift(rgb_image, max_shift=0, seed=1)
        _assert_same_image(rgb_image, same)

    def test_rgba_alpha_preserved(self, rgba_image: Image.Image) -> None:
        out = rgb_chromatic_shift(rgba_image, max_shift=12, seed=42)
        _assert_alpha_preserved(rgba_image, out)


class TestScanlineDisplacement:
    def test_determinism_and_divergence(self, rgb_image: Image.Image) -> None:
        a = scanline_displacement(rgb_image, band_height=8, max_shift=20, tear_probability=0.5, seed=10)
        b = scanline_displacement(rgb_image, band_height=8, max_shift=20, tear_probability=0.5, seed=10)
        c = scanline_displacement(rgb_image, band_height=8, max_shift=20, tear_probability=0.5, seed=11)
        _assert_same_image(a, b)
        assert not np.array_equal(np.asarray(a), np.asarray(rgb_image))
        assert not np.array_equal(np.asarray(a), np.asarray(c))

    def test_rgba_alpha_preserved(self, rgba_image: Image.Image) -> None:
        out = scanline_displacement(rgba_image, band_height=6, max_shift=15, tear_probability=0.6, seed=42)
        _assert_alpha_preserved(rgba_image, out)


class TestPixelSort:
    def test_horizontal_luminance_is_sorted(self) -> None:
        rng = np.random.default_rng(5)
        arr = rng.integers(64, 220, size=(24, 64, 3), dtype=np.uint8)
        src = Image.fromarray(arr, mode="RGB")
        out = pixel_sort(src, low_threshold=0.0, high_threshold=1.0, direction="horizontal")
        out_arr = np.asarray(out, dtype=np.float32)
        lum = (0.2126 * out_arr[:, :, 0] + 0.7152 * out_arr[:, :, 1] + 0.0722 * out_arr[:, :, 2]) / 255.0
        assert np.all(np.diff(lum, axis=1) >= -1e-6)

    def test_vertical_and_reverse(self, rgb_image: Image.Image) -> None:
        v1 = pixel_sort(rgb_image, low_threshold=0.2, high_threshold=0.85, direction="vertical", reverse=True)
        v2 = pixel_sort(rgb_image, low_threshold=0.2, high_threshold=0.85, direction="vertical", reverse=True)
        _assert_same_image(v1, v2)
        assert not np.array_equal(np.asarray(v1), np.asarray(rgb_image))

    def test_rgba_alpha_preserved(self, rgba_image: Image.Image) -> None:
        out = pixel_sort(rgba_image, low_threshold=0.2, high_threshold=0.8, direction="horizontal")
        _assert_alpha_preserved(rgba_image, out)


class TestBayerDither:
    @pytest.mark.parametrize("matrix_size", [4, 8])
    def test_quantization_levels_and_determinism(self, rgb_image: Image.Image, matrix_size: int) -> None:
        d1 = bayer_dither(rgb_image, matrix_size=matrix_size, levels=2, spread=1.0)
        d2 = bayer_dither(rgb_image, matrix_size=matrix_size, levels=2, spread=1.0)
        _assert_same_image(d1, d2)
        unique_vals = set(np.unique(np.asarray(d1)).tolist())
        assert unique_vals.issubset({0, 255})

    def test_invalid_matrix_size(self, rgb_image: Image.Image) -> None:
        with pytest.raises(ValueError):
            bayer_dither(rgb_image, matrix_size=6)

    def test_rgba_alpha_preserved(self, rgba_image: Image.Image) -> None:
        out = bayer_dither(rgba_image, matrix_size=8, levels=4)
        _assert_alpha_preserved(rgba_image, out)


class TestCrtScanlinesVignette:
    def test_scanline_darkening_and_vignette(self) -> None:
        flat = Image.fromarray(np.full((90, 120, 3), 200, dtype=np.uint8), mode="RGB")
        out = crt_scanlines_vignette(
            flat,
            scanline_spacing=3,
            scanline_strength=0.4,
            phosphor_strength=0.0,
            vignette_strength=0.5,
        )
        arr = np.asarray(out, dtype=np.float32)
        # Center bright row vs adjacent scanline row
        center_bright = arr[46, 60].mean()
        center_scanline = arr[45, 60].mean()  # 45 % 3 == 0 -> darkened
        corner = arr[1, 1].mean()
        assert center_scanline < center_bright
        assert corner < center_bright

    def test_rgba_alpha_preserved(self, rgba_image: Image.Image) -> None:
        out = crt_scanlines_vignette(rgba_image)
        _assert_alpha_preserved(rgba_image, out)


class TestJpegDatabend:
    def test_determinism_and_resilience(self, rgb_image: Image.Image) -> None:
        a = jpeg_databend(rgb_image, quality=20, corruptions=12, block_scramble=0.06, seed=77)
        b = jpeg_databend(rgb_image, quality=20, corruptions=12, block_scramble=0.06, seed=77)
        _assert_same_image(a, b)
        assert not np.array_equal(np.asarray(a), np.asarray(rgb_image))

        # Verify stability across several seeds and low quality
        for s in (1, 2, 3, 99, 1337):
            res = jpeg_databend(rgb_image, quality=10, corruptions=18, block_scramble=0.05, seed=s)
            assert res.size == rgb_image.size
            assert res.mode == "RGB"

    def test_rgba_alpha_preserved(self, rgba_image: Image.Image) -> None:
        out = jpeg_databend(rgba_image, quality=22, corruptions=8, block_scramble=0.05, seed=42)
        _assert_alpha_preserved(rgba_image, out)


class TestPresetsAndPipeline:
    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_all_presets_deterministic_and_rgba_safe(
        self, rgb_image: Image.Image, rgba_image: Image.Image, preset_name: str
    ) -> None:
        r1 = apply_preset(rgb_image, preset_name, seed=42)
        r2 = apply_preset(rgb_image, preset_name, seed=42)
        _assert_same_image(r1, r2)
        assert not np.array_equal(np.asarray(r1), np.asarray(rgb_image))

        ra = apply_preset(rgba_image, preset_name, seed=42)
        _assert_alpha_preserved(rgba_image, ra)

    def test_zero_intensity_returns_copy(self, rgb_image: Image.Image) -> None:
        out = apply_preset(rgb_image, "vhs_tracking", seed=42, intensity=0.0)
        _assert_same_image(rgb_image, out)

    def test_custom_pipeline_and_errors(self, rgb_image: Image.Image) -> None:
        steps = [
            ("bayer_dither", {"matrix_size": 4, "levels": 4}),
            ("rgb_chromatic_shift", {"max_shift": 6}),
        ]
        p1 = apply_pipeline(rgb_image, steps, seed=19)
        p2 = apply_pipeline(rgb_image, steps, seed=19)
        _assert_same_image(p1, p2)

        assert len(list_presets()) == len(PRESETS)
        with pytest.raises(ValueError):
            apply_preset(rgb_image, "does_not_exist")
        with pytest.raises(ValueError):
            apply_pipeline(rgb_image, [("unknown_op", {})])

    def test_grayscale_and_palette_inputs(self) -> None:
        gray = Image.fromarray(np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64), mode="L")
        out = apply_preset(gray, "crt_dither", seed=5)
        assert out.size == (64, 64)
        assert out.mode == "RGB"
