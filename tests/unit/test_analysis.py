from __future__ import annotations

import pytest

from vision_ai_studio.analysis import (
    caption_summary,
    labels_from_report,
    pixel_ratios,
    ranked_classes,
    token_f1,
    top_confusions,
    worst_images,
)


def _report() -> dict:
    return {
        "classes": [
            {"label": "background", "iou": 0.9, "gt_pixels": 100, "pred_pixels": 120},
            {"label": "cat", "iou": 0.4, "gt_pixels": 50, "pred_pixels": 30},
            {"label": "dog", "iou": None, "gt_pixels": 0, "pred_pixels": 7},
            {"label": "ghost", "iou": 0.0, "gt_pixels": 0, "pred_pixels": 5},
        ],
        "confusion": [
            [80, 20, 0],
            [10, 35, 5],
            [0, 7, 0],
        ],
        "per_image": [
            {"file": "a.jpg", "miou_foreground": 0.2, "pixel_accuracy": 0.5, "classes_present": 1},
            {"file": "b.jpg", "miou_foreground": 0.8, "pixel_accuracy": 0.9, "classes_present": 2},
        ],
    }


def test_ranked_classes_skips_undefined_iou() -> None:
    report = _report()
    assert labels_from_report(report) == ["background", "cat", "dog", "ghost"]
    assert [row["label"] for row in ranked_classes(report)] == ["cat", "background"]
    assert [row["label"] for row in ranked_classes(report, reverse=True)] == [
        "background",
        "cat",
    ]


def test_top_confusions_orders_off_diagonal_cells() -> None:
    pairs = top_confusions(_report(), limit=2)
    assert pairs[0] == {
        "truth": "background",
        "predicted": "cat",
        "count": 20,
        "share_of_truth": 0.2,
    }
    assert pairs[1]["truth"] == "cat"
    assert pairs[1]["predicted"] == "background"


def test_pixel_ratios_skip_classes_without_ground_truth() -> None:
    ratios = {row["label"]: row["ratio"] for row in pixel_ratios(_report())}
    assert ratios == {"background": 1.2, "cat": 0.6}


def test_worst_images_are_sorted_by_foreground_iou() -> None:
    assert [row["file"] for row in worst_images(_report())] == ["a.jpg", "b.jpg"]


def test_token_f1_bounds() -> None:
    assert token_f1("a cat on a mat", ["a cat on a mat"]) == 1.0
    assert token_f1("zebra", ["a cat"]) == 0.0
    partial = token_f1("a cat", ["a cat on a mat"])
    assert 0.0 < partial < 1.0


def test_caption_summary_flags_duplicates_and_weak_examples() -> None:
    predictions = {"1": "a cat", "2": "a cat", "3": "a dog on grass"}
    references = {
        "1": ["a cat on a mat"],
        "2": ["a cat on a mat"],
        "3": ["a dog running in a field"],
    }
    summary = caption_summary(predictions, references, limit=2)
    assert summary["images"] == 3
    assert summary["unique_captions"] == 2
    assert summary["duplicate_rate"] == pytest.approx(1 / 3)
    assert summary["mean_words"] == 8 / 3
    assert summary["scored_images"] == 3
    assert len(summary["worst"]) == 2
    assert summary["worst"][0]["token_f1"] <= summary["worst"][1]["token_f1"]
