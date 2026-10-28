from __future__ import annotations

import io

import pytest
from PIL import Image

from vision_ai_studio.image import ImageError, downscale, image_digest, load_image, to_png


def test_load_returns_rgb(scene_png: bytes) -> None:
    image = load_image(scene_png)
    assert image.mode == "RGB"
    assert image.size == (320, 240)


def test_downscale_preserves_aspect(scene_png: bytes) -> None:
    image = load_image(scene_png, max_edge=160)
    assert max(image.size) == 160
    assert round(image.width / image.height, 2) == round(320 / 240, 2)


def test_downscale_never_upscales() -> None:
    image = Image.new("RGB", (100, 50))
    assert downscale(image, 400) is image


def test_empty_bytes_rejected() -> None:
    with pytest.raises(ImageError):
        load_image(b"")


def test_corrupt_bytes_rejected() -> None:
    with pytest.raises(ImageError, match="Unsupported or corrupt"):
        load_image(b"not an image")


def test_missing_file_rejected(tmp_path) -> None:
    with pytest.raises(ImageError, match="File not found"):
        load_image(tmp_path / "nope.png")


def test_size_limit_enforced() -> None:
    noise = Image.effect_noise((600, 600), 120).convert("RGB")
    payload = to_png(noise)
    assert len(payload) > 10_000
    with pytest.raises(ImageError, match="larger than"):
        load_image(payload, max_upload_mb=0.001)


def test_exif_orientation_is_applied() -> None:
    image = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    loaded = load_image(buffer.getvalue())
    assert loaded.size == (20, 40)


def test_image_digest_is_stable(scene_png: bytes) -> None:
    assert image_digest(scene_png) == image_digest(scene_png)
    assert len(image_digest(scene_png)) == 64
    assert image_digest(scene_png) != image_digest(scene_png + b"x")
