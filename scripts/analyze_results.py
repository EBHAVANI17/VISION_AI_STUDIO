"""Turn evaluation JSON reports into an error-analysis markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vision_ai_studio.analysis import (
    caption_summary,
    pixel_ratios,
    ranked_classes,
    top_confusions,
    worst_images,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voc",
        type=Path,
        default=Path("artifacts/eval/segmentation_coco-voc.json"),
    )
    parser.add_argument("--ade", type=Path, default=Path("artifacts/eval/segmentation_ade.json"))
    parser.add_argument(
        "--caption", type=Path, default=Path("artifacts/eval/captioning_blip-base.json")
    )
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval/analysis.md"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    lines = ["# Error analysis", ""]
    found = False
    for title, path in (("COCO/VOC-21", args.voc), ("ADE20K-150", args.ade)):
        if path.is_file():
            found = True
            lines.extend(_segmentation_section(title, _load(path), args.limit))
    if args.caption.is_file():
        found = True
        lines.extend(_caption_section(_load(args.caption), args.coco_root, args.limit))
    if not found:
        print("No evaluation reports found; run the evaluation scripts first.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.output}")
    return 0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segmentation_section(title: str, report: dict[str, Any], limit: int) -> list[str]:
    lines = [
        f"## Segmentation: {title}",
        "",
        f"- model: `{report.get('version', 'unknown')}`",
        f"- images: {report.get('images')} (skipped {report.get('skipped', 0)})"
        f" at max-edge {report.get('max_edge')}",
        f"- mIoU: {_fmt(report.get('miou'))} | foreground mIoU:"
        f" {_fmt(report.get('miou_foreground'))} | pixel accuracy:"
        f" {_fmt(report.get('pixel_accuracy'))}",
        f"- generated: {report.get('generated_at', 'unknown')}",
        "",
        "### Weakest classes",
        "",
        "| class | IoU | GT pixels | predicted pixels |",
        "|---|---|---|---|",
    ]
    weakest = ranked_classes(report)[:limit]
    lines.extend(
        f"| {row['label']} | {float(row['iou']):.3f} | {int(row['gt_pixels']):,}"
        f" | {int(row['pred_pixels']):,} |"
        for row in weakest
    )
    lines.extend(["", "### Strongest classes", "", "| class | IoU |", "|---|---|"])
    strongest = ranked_classes(report, reverse=True)[:limit]
    lines.extend(f"| {row['label']} | {float(row['iou']):.3f} |" for row in strongest)

    lines.extend(
        [
            "",
            "### Most frequent confusions (ground truth -> prediction)",
            "",
            "| truth | predicted | pixels | share of truth |",
            "|---|---|---|---|",
        ]
    )
    confusions = top_confusions(report, limit=limit)
    if confusions:
        lines.extend(
            f"| {pair['truth']} | {pair['predicted']} | {pair['count']:,}"
            f" | {pair['share_of_truth']:.1%} |"
            for pair in confusions
        )
    else:
        lines.append("| — | — | — | — |")

    ratios = pixel_ratios(report)
    over = sorted(ratios, key=lambda row: float(row["ratio"]), reverse=True)[:5]
    under = sorted(ratios, key=lambda row: float(row["ratio"]))[:5]
    lines.extend(
        ["", "### Over- and under-segmentation", "", "| class | predicted / truth |", "|---|---|"]
    )
    lines.extend(f"| {row['label']} (over) | {float(row['ratio']):.2f}x |" for row in over)
    lines.extend(f"| {row['label']} (under) | {float(row['ratio']):.2f}x |" for row in under)

    lines.extend(
        [
            "",
            f"### Worst {limit} images (foreground mIoU)",
            "",
            "| image | fg mIoU | pixel accuracy | classes present |",
            "|---|---|---|---|",
        ]
    )
    lines.extend(
        f"| {row['file']} | {_fmt(row['miou_foreground'])} | {_fmt(row['pixel_accuracy'])}"
        f" | {row['classes_present']} |"
        for row in worst_images(report, limit=limit)
    )
    lines.append("")
    return lines


def _caption_section(report: dict[str, Any], coco_root: Path, limit: int) -> list[str]:
    predictions_path = report.get("predictions")
    predictions: dict[str, str] = {}
    if predictions_path and Path(predictions_path).is_file():
        predictions = _load(Path(predictions_path))
    references = _load_references(coco_root, set(predictions))
    summary = caption_summary(predictions, references, limit=limit)

    lines = [
        "## Captioning: COCO val2017",
        "",
        f"- model: `{report.get('version', 'unknown')}` | images: {report.get('images')}"
        f" | beam {report.get('beam')} | max-edge {report.get('max_edge')}",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for name, value in sorted(report.get("scores", {}).items()):
        lines.append(f"| {name} | {float(value):.4f} |")
    for name, message in report.get("errors", {}).items():
        lines.append(f"| {name} | skipped: {message[:60]} |")

    lines.extend(
        [
            "",
            "### Prediction patterns",
            "",
            f"- unique captions: {summary['unique_captions']} / {summary['images']}"
            f" (duplicate rate {summary['duplicate_rate']:.1%})",
            f"- caption length: mean {summary['mean_words']:.1f} words"
            f" (min {summary['min_words']}, max {summary['max_words']})",
        ]
    )
    if "mean_token_f1" in summary:
        lines.append(
            f"- mean token F1 vs. references: {float(summary['mean_token_f1']):.3f}"
            f" over {summary['scored_images']} images"
        )
        lines.extend(
            [
                "",
                f"### Weakest {limit} captions (token F1)",
                "",
                "| image id | caption | token F1 |",
                "|---|---|---|",
            ]
        )
        lines.extend(
            f"| {row['image_id']} | {_short(str(row['caption']), 80)}"
            f" | {float(row['token_f1']):.3f} |"
            for row in summary.get("worst", [])
        )
    else:
        lines.append("- reference captions not loaded; token F1 skipped")
    lines.append("")
    return lines


def _load_references(coco_root: Path, image_ids: set[str]) -> dict[str, list[str]]:
    path = coco_root / "annotations" / "captions_val2017.json"
    if not path.is_file() or not image_ids:
        return {}
    data = _load(path)
    captions: dict[str, list[str]] = {}
    for annotation in data.get("annotations", []):
        image_id = str(annotation["image_id"])
        if image_id in image_ids:
            captions.setdefault(image_id, []).append(str(annotation["caption"]).strip())
    return captions


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
