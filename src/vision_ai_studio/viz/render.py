"""Mask and overlay rendering with Pillow + NumPy only."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from vision_ai_studio.image import to_png
from vision_ai_studio.schemas import DetectedObject
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
    detections: Sequence[DetectedObject] = (),
) -> tuple[bytes, bytes]:
    """Return (overlay_png, mask_png)."""
    mask = align_mask(mask, image.size)
    overlay = overlay_image(image, mask, labels, alpha=alpha)
    if detections:
        overlay = draw_bounding_boxes(overlay, detections)
    return to_png(overlay), to_png(mask_image(mask, labels))


def draw_bounding_boxes(
    image: Image.Image,
    detections: Sequence[DetectedObject],
    color: tuple[int, int, int] = (255, 215, 0),
) -> Image.Image:
    """Draw bounding boxes and class tags onto the image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        xmin, ymin, xmax, ymax = det.box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        tag = f"{det.label} {det.confidence:.2f}"
        draw.text((xmin + 4, max(0, ymin - 15)), tag, fill=color)
    return annotated
