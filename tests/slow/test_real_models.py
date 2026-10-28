from __future__ import annotations

import os

import pytest
from PIL import Image

from vision_ai_studio.image import load_image
from vision_ai_studio.models.registry import get_captioner, get_segmenter
from vision_ai_studio.pipeline import Models, analyze

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("VAS_RUN_SLOW") != "1",
        reason="downloads model weights; set VAS_RUN_SLOW=1 to enable",
    ),
]


@pytest.fixture
def image(scene_png: bytes) -> Image.Image:
    return load_image(scene_png)


@pytest.mark.parametrize("segmenter_name", ["voc", "ade"])
def test_real_models_produce_results(image: Image.Image, segmenter_name: str) -> None:
    models = Models(
        segmenter=get_segmenter(segmenter_name, "cpu"),
        captioner=get_captioner("blip-base", "cpu"),
        translator=None,
    )
    result = analyze(image, models=models, target_lang="en")

    assert result.caption
    assert result.segments
    assert sum(segment.coverage for segment in result.segments) == pytest.approx(1.0)
    assert result.overlay_png.startswith(b"\x89PNG")
    assert result.image_size == image.size
    assert result.model_versions["segmenter"] != "fake"
