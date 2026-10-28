from __future__ import annotations

import io

import pytest
from PIL import Image

from vision_ai_studio.image import load_image
from vision_ai_studio.models.base import ModelInferenceError, Segmentation
from vision_ai_studio.models.fakes import FakeCaptioner, FakeSegmenter, FakeTranslator
from vision_ai_studio.pipeline import Models, analyze


class FailingCaptioner:
    name = "failing-captioner"

    def caption(self, image: Image.Image, *, beam: int = 1) -> str:
        raise RuntimeError("caption backend offline")


class FailingTranslator:
    name = "failing-translator"
    version = "failing:1"

    def translate(self, text: str, target: str) -> str:
        raise RuntimeError("translator offline")


class FailingSegmenter:
    name = "failing-segmenter"

    def predict(self, image: Image.Image) -> Segmentation:
        raise ModelInferenceError("segmenter crashed")


def fake_models(translator=None) -> Models:
    return Models(
        segmenter=FakeSegmenter(),
        captioner=FakeCaptioner(),
        translator=translator,
    )


@pytest.fixture
def image(scene_png: bytes) -> Image.Image:
    return load_image(scene_png)


def test_full_pipeline_in_english(image: Image.Image) -> None:
    result = analyze(image, models=fake_models(), target_lang="en")

    assert result.caption.startswith("a synthetic test scene")
    assert result.translated is None
    assert result.target_lang is None
    assert result.dominant_label == "person"
    assert sum(segment.coverage for segment in result.segments) == pytest.approx(1.0)
    assert result.model_versions["translator"] == "not used"
    assert {"caption", "segment", "render", "total"} <= set(result.timings_ms)
    assert result.warnings == ()


def test_pipeline_translates_when_asked(image: Image.Image) -> None:
    result = analyze(image, models=fake_models(FakeTranslator()), target_lang="hi")
    assert result.target_lang == "hi"
    assert result.translated == f"[hi] {result.caption}"
    assert result.model_versions["translator"] == "fake:translator@1"


def test_renders_match_image_size(image: Image.Image) -> None:
    result = analyze(image, models=fake_models(), overlay_alpha=0.5)
    overlay = Image.open(io.BytesIO(result.overlay_png))
    mask = Image.open(io.BytesIO(result.mask_png))
    assert overlay.size == image.size
    assert mask.size == image.size


def test_caption_failure_degrades(image: Image.Image) -> None:
    models = Models(FakeSegmenter(), FailingCaptioner(), None)
    result = analyze(image, models=models)
    assert result.caption == ""
    assert result.segments
    assert any("Captioning failed" in warning for warning in result.warnings)


def test_translation_failure_keeps_caption(image: Image.Image) -> None:
    result = analyze(
        image,
        models=fake_models(FailingTranslator()),
        target_lang="ta",
    )
    assert result.caption
    assert result.translated is None
    assert any("Translation to ta failed" in warning for warning in result.warnings)


def test_missing_translator_warns(image: Image.Image) -> None:
    result = analyze(image, models=fake_models(None), target_lang="fr")
    assert result.translated is None
    assert any("Translation is disabled" in warning for warning in result.warnings)


def test_segmentation_failure_raises(image: Image.Image) -> None:
    models = Models(FailingSegmenter(), FakeCaptioner(), None)
    with pytest.raises(ModelInferenceError):
        analyze(image, models=models)


def test_invalid_language_rejected(image: Image.Image) -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        analyze(image, models=fake_models(), target_lang="de")
