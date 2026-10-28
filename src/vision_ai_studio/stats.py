"""Turning a class-id mask into ranked, human-readable segment statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vision_ai_studio.schemas import Segment


def segment_stats(mask: np.ndarray, labels: Sequence[str]) -> tuple[Segment, ...]:
    """Aggregate a class-id mask into segments sorted by covered pixels (descending)."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if mask.size == 0:
        return ()

    ids, counts = np.unique(mask, return_counts=True)
    total = float(mask.size)
    segments = []
    for class_id, pixels in zip(ids.tolist(), counts.tolist(), strict=True):
        index = int(class_id)
        label = labels[index] if 0 <= index < len(labels) else f"class {index}"
        segments.append(
            Segment(
                class_id=index, label=str(label), pixels=int(pixels), coverage=float(pixels) / total
            )
        )
    segments.sort(key=lambda segment: (-segment.pixels, segment.class_id))
    return tuple(segments)


def dominant_label(segments: Sequence[Segment], *, ignore: Sequence[str] = ("background",)) -> str:
    """Largest segment, skipping common 'background' labels when anything else exists."""
    if not segments:
        return "unknown"
    ignored = {name.lower() for name in ignore}
    for segment in segments:
        if segment.label.lower() not in ignored:
            return segment.label
    return segments[0].label
