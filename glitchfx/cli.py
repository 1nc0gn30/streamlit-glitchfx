"""Zero-Streamlit headless CLI for deterministic image glitch processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image

from glitchfx.engine import PRESETS, apply_preset, list_presets
from glitchfx.samples import SAMPLE_STYLES, generate_sample_image

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})


def _parse_size(raw: str) -> tuple[int, int]:
    parts = raw.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise argparse.ArgumentTypeError(f"Invalid size '{raw}'. Expected WIDTHxHEIGHT (e.g. 800x520).")
    w, h = int(parts[0]), int(parts[1])
    if w < 16 or h < 16:
        raise argparse.ArgumentTypeError("Sample dimensions must be at least 16x16.")
    return w, h


def _collect_inputs(paths: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {raw}")
        if p.is_dir():
            dir_images = sorted(
                child
                for child in p.iterdir()
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            resolved.extend(dir_images)
        else:
            resolved.append(p)
    return resolved


def _save_image(image: Image.Image, target: Path, fmt: str | None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = (fmt or target.suffix.lstrip(".") or "png").lower()
    if ext in ("jpg", "jpeg"):
        image.convert("RGB").save(target, format="JPEG", quality=92)
    elif ext == "webp":
        image.save(target, format="WEBP", lossless=True, quality=100)
    else:
        image.save(target, format="PNG")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glitchfx",
        description="Deterministic image glitch-art processor (Pillow + NumPy, zero Streamlit required).",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input image file(s) or directory to process.",
    )
    parser.add_argument(
        "--preset",
        default="vhs_tracking",
        help="Glitch preset name (default: vhs_tracking). See --list-presets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic integer seed (default: 42).",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        help="Effect intensity multiplier >= 0.0 (default: 1.0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path (single-image mode) or output directory (batch mode).",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("png", "webp", "jpg"),
        default=None,
        help="Force output image format (default: inferred from --out extension, else png).",
    )
    parser.add_argument(
        "--sample",
        choices=tuple(SAMPLE_STYLES.keys()),
        default=None,
        help="Generate procedural sample artwork instead of reading an input file.",
    )
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=(800, 520),
        help="Dimensions WIDTHxHEIGHT when using --sample (default: 800x520).",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all built-in presets and exit.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        presets = list_presets()
        if args.as_json:
            payload = [
                {
                    "name": spec.name,
                    "label": spec.label,
                    "description": spec.description,
                    "steps": [step_name for step_name, _ in spec.steps],
                }
                for spec in presets
            ]
            print(json.dumps(payload, indent=2))
        else:
            for spec in presets:
                steps_str = " -> ".join(step_name for step_name, _ in spec.steps)
                print(f"{spec.name:<18} {spec.label:<26} | {spec.description} [{steps_str}]")
        return 0

    if args.preset not in PRESETS:
        valid = ", ".join(PRESETS.keys())
        print(f"error: unknown preset '{args.preset}'. Available: {valid}", file=sys.stderr)
        return 2

    if args.intensity < 0.0:
        print("error: --intensity must be >= 0.0", file=sys.stderr)
        return 2

    # Procedural sample mode
    if args.sample is not None:
        if args.inputs:
            print("error: cannot combine positional input paths with --sample", file=sys.stderr)
            return 2
        w, h = args.size
        src = generate_sample_image(args.sample, width=w, height=h, seed=args.seed)
        result = apply_preset(src, args.preset, seed=args.seed, intensity=args.intensity)
        ext = args.output_format or (args.out.suffix.lstrip(".") if args.out else "png") or "png"
        out_path = args.out or Path(f"{args.sample}_{args.preset}_s{args.seed}.{ext}")
        saved = _save_image(result, out_path, args.output_format)
        record = {
            "source": f"sample:{args.sample}",
            "output": str(saved),
            "preset": args.preset,
            "seed": args.seed,
            "size": list(result.size),
            "mode": result.mode,
        }
        if args.as_json:
            print(json.dumps(record, indent=2))
        else:
            print(f"Wrote {saved} ({result.size[0]}x{result.size[1]} {result.mode}, preset={args.preset}, seed={args.seed})")
        return 0

    # File / batch mode
    if not args.inputs:
        parser.print_usage(sys.stderr)
        print("error: provide one or more input images/directories, or pass --sample / --list-presets", file=sys.stderr)
        return 2

    try:
        input_files = _collect_inputs(args.inputs)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not input_files:
        print("error: no supported image files found in the given input path(s).", file=sys.stderr)
        return 2

    is_batch = len(input_files) > 1 or (args.out is not None and args.out.is_dir())
    records: list[dict[str, object]] = []

    for idx, src_path in enumerate(input_files):
        with Image.open(src_path) as opened:
            opened.load()
            img_seed = args.seed if not is_batch else args.seed + idx
            result = apply_preset(opened, args.preset, seed=img_seed, intensity=args.intensity)

        ext = args.output_format or (
            args.out.suffix.lstrip(".") if (args.out is not None and not is_batch and args.out.suffix) else "png"
        )
        if not is_batch and args.out is not None:
            out_path = args.out
        else:
            out_dir = args.out if (args.out is not None and is_batch) else src_path.parent
            out_path = out_dir / f"{src_path.stem}_{args.preset}.{ext}"

        saved = _save_image(result, out_path, args.output_format)
        records.append(
            {
                "source": str(src_path),
                "output": str(saved),
                "preset": args.preset,
                "seed": img_seed,
                "size": list(result.size),
                "mode": result.mode,
            }
        )
        if not args.as_json:
            print(f"Wrote {saved} ({result.size[0]}x{result.size[1]} {result.mode}, preset={args.preset}, seed={img_seed})")

    if args.as_json:
        print(json.dumps(records if is_batch else records[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
