"""Deterministic, label-stable colors for segmentation classes."""

from __future__ import annotations

import colorsys
import hashlib
from collections.abc import Sequence

import numpy as np

BACKGROUND_RGB = (96, 100, 110)


def class_color(class_id: int, label: str) -> tuple[int, int, int]:
    """Stable color per (id, label) pair; identical across runs and platforms."""
    if label.strip().strip("_").lower() in {"background", "unlabeled", "ignore", "other"}:
        return BACKGROUND_RGB

    digest = hashlib.blake2b(f"{class_id}:{label}".encode(), digest_size=8).digest()
    seed = int.from_bytes(digest, "big")
    hue = (seed % 3600) / 3600.0
    saturation = 0.52 + ((seed >> 12) % 22) / 100.0
    lightness = 0.46 + ((seed >> 24) % 14) / 100.0
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (round(red * 255), round(green * 255), round(blue * 255))


def hex_color(class_id: int, label: str) -> str:
    red, green, blue = class_color(class_id, label)
    return f"#{red:02x}{green:02x}{blue:02x}"


def colorize_mask(mask: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Map a class-id mask to an RGB array using the class palette."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

    clipped = np.clip(mask, 0, None).astype(np.int32, copy=False)
    max_id = int(clipped.max()) if clipped.size else 0
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for class_id in np.unique(clipped).tolist():
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class {class_id}"
        lut[class_id] = class_color(int(class_id), str(label))
    return np.asarray(lut[clipped], dtype=np.uint8)


def label_for(class_id: int, labels: Sequence[str]) -> str:
    if 0 <= class_id < len(labels):
        return str(labels[class_id])
    return f"class {class_id}"
