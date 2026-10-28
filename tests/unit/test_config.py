from __future__ import annotations

import pytest

from vision_ai_studio.config import Settings


def test_defaults() -> None:
    settings = Settings.from_env({})
    assert settings.segmenter == "ade"
    assert settings.captioner == "blip-base"
    assert settings.translator == "nllb"
    assert settings.models == "real"
    assert settings.overlay_alpha == 0.45


def test_env_overrides() -> None:
    settings = Settings.from_env(
        {
            "VAS_SEGMENTER": "VOC",
            "VAS_CAPTIONER": "blip-large",
            "VAS_TRANSLATOR": "none",
            "VAS_MODELS": "fake",
            "VAS_MAX_EDGE": "1024",
            "VAS_OVERLAY_ALPHA": "0.3",
            "VAS_LORA_REPO": "user/adapter",
        }
    )
    assert settings.segmenter == "voc"
    assert settings.captioner == "blip-large"
    assert settings.translator == "none"
    assert settings.models == "fake"
    assert settings.max_edge == 1024
    assert settings.overlay_alpha == 0.3
    assert settings.lora_repo == "user/adapter"


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown segmenter"):
        Settings(segmenter="nope").validate()
    with pytest.raises(ValueError, match="overlay_alpha"):
        Settings(overlay_alpha=2.0).validate()
    with pytest.raises(ValueError, match="VAS_MAX_EDGE"):
        Settings.from_env({"VAS_MAX_EDGE": "big"})


def test_cache_key_tracks_models_not_layout() -> None:
    base = Settings()
    assert base.cache_key() == Settings(overlay_alpha=0.2).cache_key()
    assert base.cache_key() != Settings(segmenter="voc").cache_key()
    assert base.cache_key() != Settings(lora_repo="user/adapter").cache_key()


def test_with_overrides_validates() -> None:
    with pytest.raises(ValueError):
        Settings().with_overrides(captioner="unknown")
    updated = Settings().with_overrides(segmenter="voc", overlay_alpha=0.6)
    assert updated.segmenter == "voc"
    assert updated.overlay_alpha == 0.6
