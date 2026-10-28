"""Deterministic fakes used by tests, CI and UI development without model weights."""

from __future__ import annotations

import numpy as np
from PIL import Image

from vision_ai_studio.models.base import Segmentation


class FakeSegmenter:
    name = "fake-segmenter"
    version = "fake:segmentation@1"
    labels = ("background", "person", "chair")

    def predict(self, image: Image.Image) -> Segmentation:
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.int32)

        left, right = int(width * 0.30), int(width * 0.70)
        top, bottom = int(height * 0.15), int(height * 0.92)
        mask[top:bottom, left:right] = 1

        chair_left, chair_top = int(width * 0.70), int(height * 0.55)
        mask[chair_top:, chair_left:] = 2
        return Segmentation(mask=mask, labels=self.labels, version=self.version)


class FakeCaptioner:
    name = "fake-captioner"
    version = "fake:captioner@1"

    def caption(self, image: Image.Image, *, beam: int = 1) -> str:
        del image, beam
        return "a synthetic test scene with a person next to a chair"


class FakeTranslator:
    name = "fake-translator"
    version = "fake:translator@1"

    def translate(self, text: str, target: str) -> str:
        return f"[{target}] {text}"


class FakeSynthesizer:
    name = "fake-tts"
    version = "fake:tts@1"

    def speak(self, text: str, lang: str) -> bytes:
        del text, lang
        return b""
