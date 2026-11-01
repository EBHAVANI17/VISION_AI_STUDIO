"""Error-analysis helpers for segmentation and captioning evaluation reports.

The evaluation scripts emit aggregate metrics plus a class confusion matrix and per-image
scores. This module turns those into the views needed to find failure patterns: weak classes,
systematic confusions, over/under-prediction and the worst images.
"""

from __future__ import annotations

from typing import Any


def labels_from_report(report: dict[str, Any]) -> list[str]:
    return [str(row["label"]) for row in report.get("classes", [])]


def ranked_classes(report: dict[str, Any], *, reverse: bool = False) -> list[dict[str, Any]]:
    """Classes with IoU and ground truth, sorted weakest-first by default."""
    rows = [
        row
        for row in report.get("classes", [])
        if row.get("iou") is not None and int(row.get("gt_pixels") or 0) > 0
    ]
    return sorted(rows, key=lambda row: float(row["iou"]), reverse=reverse)


def top_confusions(report: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    """Largest off-diagonal cells: ground-truth class predicted as another class."""
    matrix = report.get("confusion") or []
    labels = labels_from_report(report)
    pairs: list[dict[str, Any]] = []
    for truth_index, row in enumerate(matrix):
        total = sum(int(count) for count in row)
        for predicted_index, count in enumerate(row):
            if truth_index == predicted_index or not count:
                continue
            pairs.append(
                {
                    "truth": labels[truth_index],
                    "predicted": labels[predicted_index],
                    "count": int(count),
                    "share_of_truth": int(count) / total if total else 0.0,
                }
            )
    return sorted(pairs, key=lambda pair: int(pair["count"]), reverse=True)[:limit]


def pixel_ratios(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Predicted/ground-truth pixel ratio per class (over- and under-segmentation)."""
    rows: list[dict[str, Any]] = []
    for row in report.get("classes", []):
        truth = int(row.get("gt_pixels") or 0)
        predicted = int(row.get("pred_pixels") or 0)
        if truth <= 0:
            continue
        rows.append(
            {
                "label": str(row["label"]),
                "gt_pixels": truth,
                "pred_pixels": predicted,
                "ratio": predicted / truth,
            }
        )
    return rows


def worst_images(report: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    rows = [row for row in report.get("per_image", []) if row.get("miou_foreground") is not None]
    return sorted(rows, key=lambda row: float(row["miou_foreground"]))[:limit]


def token_f1(prediction: str, references: list[str]) -> float:
    """Best bag-of-words F1 against any single reference (a cheap BLEU proxy)."""
    predicted = prediction.lower().split()
    if not predicted:
        return 0.0
    predicted_tokens = set(predicted)
    best = 0.0
    for reference in references:
        reference_tokens = set(reference.lower().split())
        if not reference_tokens:
            continue
        overlap = len(predicted_tokens & reference_tokens)
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(reference_tokens)
        if precision + recall:
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def caption_summary(
    predictions: dict[str, str],
    references: dict[str, list[str]] | None = None,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    captions = [caption for caption in predictions.values() if caption.strip()]
    lengths = [len(caption.split()) for caption in captions]
    unique = len(set(captions))
    summary: dict[str, Any] = {
        "images": len(captions),
        "unique_captions": unique,
        "duplicate_rate": 1 - unique / len(captions) if captions else 0.0,
        "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
    }
    if references:
        scored: list[dict[str, Any]] = []
        for image_id, caption in predictions.items():
            refs = references.get(str(image_id))
            if not refs:
                continue
            scored.append(
                {
                    "image_id": str(image_id),
                    "caption": caption,
                    "token_f1": token_f1(caption, refs),
                }
            )
        summary["scored_images"] = len(scored)
        summary["mean_token_f1"] = (
            sum(float(row["token_f1"]) for row in scored) / len(scored) if scored else 0.0
        )
        summary["worst"] = sorted(scored, key=lambda row: float(row["token_f1"]))[:limit]
    return summary
