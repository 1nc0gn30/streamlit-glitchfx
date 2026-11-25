"""End-to-end CLI and app.py integrity tests."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from glitchfx.engine import PRESETS
from glitchfx.samples import generate_sample_image

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "glitchfx", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_list_presets_text_and_json() -> None:
    res = _run_cli("--list-presets")
    assert res.returncode == 0
    for name in PRESETS:
        assert name in res.stdout

    res_json = _run_cli("--list-presets", "--json")
    assert res_json.returncode == 0
    parsed = json.loads(res_json.stdout)
    assert {item["name"] for item in parsed} == set(PRESETS.keys())


def test_cli_single_image_determinism_and_formats(tmp_path: Path) -> None:
    src_path = tmp_path / "source.png"
    generate_sample_image("bauhaus_prism", width=160, height=120, seed=3).save(src_path)

    out1 = tmp_path / "out1.png"
    out2 = tmp_path / "out2.png"
    out_webp = tmp_path / "out.webp"

    r1 = _run_cli(str(src_path), "--preset", "vhs_tracking", "--seed", "77", "--out", str(out1))
    r2 = _run_cli(str(src_path), "--preset", "vhs_tracking", "--seed", "77", "--out", str(out2))
    r3 = _run_cli(str(src_path), "--preset", "crt_dither", "--seed", "77", "--out", str(out_webp))

    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    assert r3.returncode == 0, r3.stderr
    assert out1.read_bytes() == out2.read_bytes()

    with Image.open(out_webp) as im:
        assert im.size == (160, 120)
        assert im.format == "WEBP"


def test_cli_sample_mode(tmp_path: Path) -> None:
    out_path = tmp_path / "sample_glitched.png"
    res = _run_cli(
        "--sample",
        "signal_testcard",
        "--size",
        "192x128",
        "--preset",
        "pixel_melt",
        "--seed",
        "15",
        "--out",
        str(out_path),
        "--json",
    )
    assert res.returncode == 0, res.stderr
    meta = json.loads(res.stdout)
    assert meta["size"] == [192, 128]
    assert meta["preset"] == "pixel_melt"
    assert out_path.exists()


def test_cli_batch_directory(tmp_path: Path) -> None:
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir()
    out_dir.mkdir()

    for idx, style in enumerate(("neon_grid", "bauhaus_prism")):
        generate_sample_image(style, width=120, height=90, seed=idx).save(in_dir / f"img_{idx}.png")

    res = _run_cli(
        str(in_dir),
        "--preset",
        "databend_raw",
        "--seed",
        "100",
        "--out",
        str(out_dir),
        "--format",
        "webp",
        "--json",
    )
    assert res.returncode == 0, res.stderr
    records = json.loads(res.stdout)
    assert len(records) == 2
    for rec in records:
        p = Path(rec["output"])
        assert p.exists()
        assert p.suffix == ".webp"
        with Image.open(p) as im:
            assert im.size == (120, 90)
            assert np.asarray(im).shape == (90, 120, 3)


def test_cli_error_cases(tmp_path: Path) -> None:
    bad_preset = _run_cli("--sample", "neon_grid", "--preset", "invalid_preset")
    assert bad_preset.returncode != 0
    assert "unknown preset" in bad_preset.stderr.lower()

    missing = _run_cli(str(tmp_path / "missing.png"))
    assert missing.returncode != 0


def test_app_py_valid_syntax() -> None:
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source, filename="app.py")
