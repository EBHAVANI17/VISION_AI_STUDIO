"""Image loading, validation and normalization."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 40_000_000  # ~40 MP; larger inputs are rejected before any model runs
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageError(ValueError):
    """Raised when user-supplied image data cannot be used."""


def load_image(
    source: bytes | bytearray | str | Path,
    *,
    max_upload_mb: int = 15,
    max_edge: int = 2048,
) -> Image.Image:
    """Decode, orient, convert to RGB and bound the size of an image."""
    raw = _read_bytes(source)
    if not raw:
        raise ImageError("The uploaded file is empty.")

    limit = max_upload_mb * 1024 * 1024
    if len(raw) > limit:
        raise ImageError(
            f"The image is larger than {max_upload_mb} MB. Please upload a smaller file."
        )

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            width, height = opened.size
            if width * height > MAX_PIXELS:
                raise ImageError(
                    f"The image has {width}x{height} pixels; the limit is {MAX_PIXELS:,}."
                )
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except ImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageError("The image is too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError("Unsupported or corrupt image file. Use JPEG, PNG or WebP.") from exc

    return downscale(image, max_edge)


def downscale(image: Image.Image, max_edge: int) -> Image.Image:
    """Shrink so that the longest edge is at most ``max_edge`` (never upscales)."""
    longest = max(image.size)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bytes(source: bytes | bytearray | str | Path) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ImageError(f"File not found: {path}")
        return path.read_bytes()
    raise ImageError(f"Unsupported image source type: {type(source).__name__}")
