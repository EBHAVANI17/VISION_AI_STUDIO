"""Runtime configuration, sourced from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

SEGMENTERS: dict[str, str] = {
    "ade": "SegFormer-B2 - ADE20K (150 classes)",
    "voc": "DeepLabV3-R50 - COCO/VOC (21 classes)",
}

CAPTIONERS: dict[str, str] = {
    "blip-base": "BLIP base (fast)",
    "blip-large": "BLIP large (better, slower on CPU)",
}

TRANSLATORS: dict[str, str] = {
    "nllb": "NLLB-200 distilled 600M (offline)",
    "google": "Google Translate (web, may rate-limit)",
    "none": "Disabled",
}

MODES: dict[str, str] = {
    "real": "Pretrained models",
    "fake": "Deterministic fakes (tests / UI development)",
}


@dataclass(frozen=True)
class Settings:
    segmenter: str = "ade"
    captioner: str = "blip-base"
    translator: str = "nllb"
    device: str = "auto"
    models: str = "real"
    lora_repo: str | None = None
    caption_revision: str | None = None
    segmenter_revision: str | None = None
    max_edge: int = 2048
    max_upload_mb: int = 15
    overlay_alpha: float = 0.45

    def validate(self) -> None:
        if self.segmenter not in SEGMENTERS:
            raise ValueError(
                f"Unknown segmenter {self.segmenter!r}; choose one of {sorted(SEGMENTERS)}"
            )
        if self.captioner not in CAPTIONERS:
            raise ValueError(
                f"Unknown captioner {self.captioner!r}; choose one of {sorted(CAPTIONERS)}"
            )
        if self.translator not in TRANSLATORS:
            raise ValueError(
                f"Unknown translator {self.translator!r}; choose one of {sorted(TRANSLATORS)}"
            )
        if self.models not in MODES:
            raise ValueError(f"Unknown model mode {self.models!r}; choose one of {sorted(MODES)}")
        if not 256 <= self.max_edge <= 4096:
            raise ValueError("max_edge must be between 256 and 4096")
        if not 1 <= self.max_upload_mb <= 100:
            raise ValueError("max_upload_mb must be between 1 and 100")
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("overlay_alpha must be between 0 and 1")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        settings = cls(
            segmenter=env.get("VAS_SEGMENTER", "ade").strip().lower(),
            captioner=env.get("VAS_CAPTIONER", "blip-base").strip().lower(),
            translator=env.get("VAS_TRANSLATOR", "nllb").strip().lower(),
            device=env.get("VAS_DEVICE", "auto").strip(),
            models=env.get("VAS_MODELS", "real").strip().lower(),
            lora_repo=env.get("VAS_LORA_REPO") or None,
            caption_revision=env.get("VAS_CAPTION_REVISION") or None,
            segmenter_revision=env.get("VAS_SEGMENTER_REVISION") or None,
            max_edge=_int_env(env, "VAS_MAX_EDGE", 2048),
            max_upload_mb=_int_env(env, "VAS_MAX_UPLOAD_MB", 15),
            overlay_alpha=_float_env(env, "VAS_OVERLAY_ALPHA", 0.45),
        )
        settings.validate()
        return settings

    def with_overrides(self, **overrides: object) -> Settings:
        updated = replace(self, **overrides)  # type: ignore[arg-type]
        updated.validate()
        return updated

    def cache_key(self) -> tuple[object, ...]:
        """Hashable identity of the model-affecting parts of the config."""
        return (
            self.segmenter,
            self.captioner,
            self.translator,
            self.device,
            self.models,
            self.lora_repo,
            self.caption_revision,
            self.segmenter_revision,
        )


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc
