"""Model interfaces shared by real implementations and test fakes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded (missing extras, no weights, ...)."""


class ModelInferenceError(RuntimeError):
    """Raised when a loaded model fails on an input."""


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray  # HxW int32 class ids, original image resolution
    labels: tuple[str, ...]  # class id -> display label
    version: str


class Segmenter(Protocol):
    name: str
    version: str

    def predict(self, image: Image.Image) -> Segmentation: ...


class Captioner(Protocol):
    name: str
    version: str

    def caption(self, image: Image.Image, *, beam: int = 1) -> str: ...


class Translator(Protocol):
    name: str
    version: str

    def translate(self, text: str, target: str) -> str: ...


class ObjectDetector(Protocol):
    name: str
    version: str

    def detect(self, image: Image.Image, *, threshold: float = 0.25) -> list[Any]: ...
