"""Latency benchmark for the full pipeline (caption + segment + render)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

from PIL import Image

from vision_ai_studio.config import Settings
from vision_ai_studio.image import load_image
from vision_ai_studio.pipeline import analyze, build_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("tests/fixtures/sample.jpg"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seg", choices=["ade", "voc"], default="ade")
    parser.add_argument("--caption", choices=["blip-base", "blip-large"], default="blip-base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args(argv)

    settings = Settings(
        segmenter=args.seg,
        captioner=args.caption,
        translator="none",
        device=args.device,
    )
    base = load_image(args.image, max_edge=max(args.sizes))
    models = build_models(settings)
    print(f"segmenter: {models.segmenter.version}")
    print(f"captioner: {getattr(models.captioner, 'version', 'unknown')}")
    print()

    results = []
    for size in args.sizes:
        image = _fit(base, size)
        analyze(image, models=models)  # warm-up: excludes lazy model loading

        totals: list[float] = []
        stages: dict[str, list[float]] = {}
        for _ in range(args.runs):
            started = perf_counter()
            result = analyze(image, models=models)
            totals.append((perf_counter() - started) * 1000)
            for name, value in result.timings_ms.items():
                stages.setdefault(name, []).append(value)

        row = {
            "size": f"{image.width}x{image.height}",
            "runs": args.runs,
            "total_p50_ms": round(statistics.median(totals), 1),
            "total_p95_ms": round(_percentile(totals, 0.95), 1),
            "stages_p50_ms": {
                name: round(statistics.median(values), 1) for name, values in stages.items()
            },
        }
        results.append(row)
        print(
            f"{row['size']:>10}  total p50 {row['total_p50_ms']:>7.0f} ms"
            f"  p95 {row['total_p95_ms']:>7.0f} ms  "
            + " ".join(f"{name}={value:.0f}" for name, value in row["stages_p50_ms"].items())
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


def _fit(image: Image.Image, max_edge: int) -> Image.Image:
    longest = max(image.size)
    if longest == max_edge:
        return image
    scale = max_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
