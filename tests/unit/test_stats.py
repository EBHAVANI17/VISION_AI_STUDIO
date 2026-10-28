from __future__ import annotations

import numpy as np
import pytest

from vision_ai_studio.stats import dominant_label, segment_stats


def test_counts_and_coverage() -> None:
    mask = np.array([[0, 0, 1], [2, 1, 1]], dtype=np.int32)
    segments = segment_stats(mask, ["background", "person", "chair"])

    assert [segment.label for segment in segments] == ["person", "background", "chair"]
    assert [segment.pixels for segment in segments] == [3, 2, 1]
    assert sum(segment.coverage for segment in segments) == pytest.approx(1.0)
    assert segments[0].coverage == pytest.approx(0.5)


def test_unknown_ids_get_fallback_labels() -> None:
    mask = np.full((2, 2), 7, dtype=np.int32)
    segments = segment_stats(mask, ["background"])
    assert segments[0].label == "class 7"
    assert segments[0].coverage == 1.0


def test_empty_mask() -> None:
    assert segment_stats(np.zeros((0, 0), dtype=np.int32), []) == ()


def test_non_2d_rejected() -> None:
    with pytest.raises(ValueError, match="2D"):
        segment_stats(np.zeros((2, 2, 3), dtype=np.int32), [])


def test_dominant_skips_background() -> None:
    mask = np.array([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=np.int32)
    segments = segment_stats(mask, ["background", "dog"])
    assert segments[0].label == "background"
    assert dominant_label(segments) == "dog"


def test_dominant_falls_back_to_background() -> None:
    mask = np.zeros((2, 2), dtype=np.int32)
    segments = segment_stats(mask, ["background"])
    assert dominant_label(segments) == "background"
    assert dominant_label(()) == "unknown"
