from __future__ import annotations

import pytest

from vision_ai_studio.models.registry import LazyTranslator
from vision_ai_studio.models.translator import (
    LANGUAGES,
    NLLB_CODES,
    GoogleTranslator,
    NullTranslator,
    build_translator,
    normalize_lang,
)


def test_normalize_lang_variants() -> None:
    assert normalize_lang("EN") == "en"
    assert normalize_lang("zh-CN") == "zh"
    assert normalize_lang("zh_CN") == "zh"
    assert normalize_lang(" hi ") == "hi"
    with pytest.raises(ValueError, match="Unsupported language"):
        normalize_lang("de")


def test_every_language_has_an_nllb_code() -> None:
    assert set(NLLB_CODES) == set(LANGUAGES)


def test_null_translator_returns_input() -> None:
    translator = NullTranslator()
    assert translator.translate("hello", "hi") == "hello"
    assert translator.version == "none"
    with pytest.raises(ValueError):
        translator.translate("hello", "de")


def test_factories() -> None:
    assert isinstance(build_translator("none"), NullTranslator)
    assert isinstance(build_translator("google"), GoogleTranslator)
    with pytest.raises(ValueError):
        build_translator("babel")


def test_lazy_translator_defers_until_first_use() -> None:
    lazy = LazyTranslator("nllb", device="cpu", mode="fake")
    assert lazy.version == "not loaded"
    assert lazy.translate("hello", "hi") == "[hi] hello"
    assert lazy.version == "fake:translator@1"
