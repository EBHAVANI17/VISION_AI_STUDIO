"""Mask and overlay rendering with Pillow + NumPy only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from vision_ai_studio.image import to_png
from vision_ai_studio.viz.palette import colorize_mask


def align_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a mask to ``size`` (width, height) if a model returned a different resolution."""
    width, height = size
    if mask.shape == (height, width):
        return mask
    resized = Image.fromarray(mask.astype(np.int32, copy=False))
    resized = resized.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.int32)


def mask_image(mask: np.ndarray, labels: Sequence[str]) -> Image.Image:
    return Image.fromarray(colorize_mask(mask, labels))


def overlay_image(
    image: Image.Image,
    mask: np.ndarray,
    labels: Sequence[str],
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend the colorized mask over the original image."""
    alpha = min(max(alpha, 0.0), 1.0)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = colorize_mask(mask, labels).astype(np.float32)
    blended = base * (1.0 - alpha) + colors * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def render_pngs(
    image: Image.Image,
    mask: np.ndarray,
    labels: Sequence[str],
    *,
    alpha: float = 0.45,
) -> tuple[bytes, bytes]:
    """Return (overlay_png, mask_png)."""
    mask = align_mask(mask, image.size)
    return to_png(overlay_image(image, mask, labels, alpha=alpha)), to_png(mask_image(mask, labels))
