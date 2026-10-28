"""Shared fixtures: tiny synthetic images, no network required."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"


@pytest.fixture
def scene_png() -> bytes:
    image = Image.new("RGB", (320, 240), (28, 58, 92))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 140, 200], fill=(220, 80, 60))
    draw.ellipse([180, 80, 300, 220], fill=(90, 200, 120))
    draw.line([0, 230, 320, 210], fill=(240, 240, 240), width=4)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def scene_path(tmp_path: Path, scene_png: bytes) -> Path:
    path = tmp_path / "scene.png"
    path.write_bytes(scene_png)
    return path
