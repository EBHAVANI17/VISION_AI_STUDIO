from __future__ import annotations

import numpy as np
from PIL import Image

from vision_ai_studio.viz.palette import (
    BACKGROUND_RGB,
    class_color,
    colorize_mask,
    hex_color,
    label_for,
)
from vision_ai_studio.viz.render import align_mask, overlay_image, render_pngs


def test_colors_are_deterministic_and_valid() -> None:
    first = class_color(3, "dog")
    assert first == class_color(3, "dog")
    assert all(0 <= channel <= 255 for channel in first)
    assert class_color(3, "dog") != class_color(4, "cat")


def test_background_is_neutral() -> None:
    assert class_color(0, "background") == BACKGROUND_RGB
    assert class_color(0, "BACKGROUND") == BACKGROUND_RGB
    assert hex_color(0, "background") == "#60646e"


def test_colorize_shapes_and_values() -> None:
    mask = np.array([[0, 1], [1, 2]], dtype=np.int32)
    rgb = colorize_mask(mask, ["background", "person", "chair"])
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert tuple(rgb[0, 1]) == class_color(1, "person")


def test_colorize_handles_ids_beyond_labels() -> None:
    rgb = colorize_mask(np.array([[9]], dtype=np.int32), ["background"])
    assert tuple(rgb[0, 0]) == class_color(9, "class 9")
    assert label_for(9, ["background"]) == "class 9"


def test_align_mask_resizes() -> None:
    mask = np.zeros((2, 2), dtype=np.int32)
    aligned = align_mask(mask, (8, 4))
    assert aligned.shape == (4, 8)


def test_overlay_alpha_extremes() -> None:
    image = Image.new("RGB", (2, 2), (10, 20, 30))
    mask = np.ones((2, 2), dtype=np.int32)
    labels = ["background", "person"]

    transparent = np.asarray(overlay_image(image, mask, labels, alpha=0.0))
    assert (transparent == (10, 20, 30)).all()

    opaque = np.asarray(overlay_image(image, mask, labels, alpha=1.0))
    assert tuple(opaque[0, 0]) == class_color(1, "person")


def test_render_pngs_are_pngs() -> None:
    image = Image.new("RGB", (16, 8), (0, 0, 0))
    mask = np.zeros((8, 16), dtype=np.int32)
    overlay, mask_png = render_pngs(image, mask, ["background"], alpha=0.4)
    assert overlay.startswith(b"\x89PNG")
    assert mask_png.startswith(b"\x89PNG")
