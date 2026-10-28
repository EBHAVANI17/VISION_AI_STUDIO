"""Lazy, process-wide model singletons.

Nothing here imports torch at module import time: fake mode and unit tests stay fast,
and the CLI only pays for the models it actually uses.
"""

from __future__ import annotations

import os
from functools import lru_cache

from vision_ai_studio.models.base import Captioner, Segmenter, Translator


def resolve_device(spec: str = "auto") -> str:
    import torch

    _configure_threads(torch)
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if spec == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")
    return spec


def _configure_threads(torch_module) -> None:
    cores = os.cpu_count() or 1
    torch_module.set_num_threads(max(1, min(4, cores)))


@lru_cache(maxsize=8)
def get_segmenter(
    name: str, device: str, revision: str | None = None, mode: str = "real"
) -> Segmenter:
    if mode == "fake":
        from vision_ai_studio.models.fakes import FakeSegmenter

        return FakeSegmenter()
    from vision_ai_studio.models.segmenter import build_segmenter

    return build_segmenter(name, device=device, revision=revision)


@lru_cache(maxsize=8)
def get_captioner(
    variant: str,
    device: str,
    revision: str | None = None,
    lora_repo: str | None = None,
    mode: str = "real",
) -> Captioner:
    if mode == "fake":
        from vision_ai_studio.models.fakes import FakeCaptioner

        return FakeCaptioner()
    from vision_ai_studio.models.captioner import build_captioner

    return build_captioner(variant, device=device, revision=revision, lora_repo=lora_repo)


@lru_cache(maxsize=4)
def get_translator(
    name: str, device: str = "cpu", revision: str | None = None, mode: str = "real"
) -> Translator:
    if mode == "fake":
        from vision_ai_studio.models.fakes import FakeTranslator

        return FakeTranslator()
    from vision_ai_studio.models.translator import build_translator

    return build_translator(name, device=device, revision=revision)


class LazyTranslator:
    """Defers the (often multi-GB) translator download until the first non-English request."""

    name = "lazy"
    version = "not loaded"

    def __init__(self, backend: str, *, device: str, mode: str) -> None:
        self._backend = backend
        self._device = device
        self._mode = mode
        self._inner: Translator | None = None

    def translate(self, text: str, target: str) -> str:
        if self._inner is None:
            self._inner = get_translator(self._backend, self._device, None, self._mode)
            self.version = str(getattr(self._inner, "version", self._backend))
        return self._inner.translate(text, target)


def clear_model_cache() -> None:
    get_segmenter.cache_clear()
    get_captioner.cache_clear()
    get_translator.cache_clear()
