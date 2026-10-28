"""Serializable result types shared by the UI, the CLI and the eval scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    class_id: int
    label: str
    pixels: int
    coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "pixels": self.pixels,
            "coverage": round(self.coverage, 6),
        }


@dataclass(frozen=True)
class AnalysisResult:
    caption: str
    segments: tuple[Segment, ...]
    dominant_label: str
    image_size: tuple[int, int]
    overlay_png: bytes
    mask_png: bytes
    translated: str | None = None
    target_lang: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def dominant_segment(self) -> Segment | None:
        return self.segments[0] if self.segments else None

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe view: no binary artifacts."""
        return {
            "caption": self.caption,
            "translated": self.translated,
            "target_lang": self.target_lang,
            "dominant_label": self.dominant_label,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "segments": [segment.to_dict() for segment in self.segments],
            "timings_ms": {key: round(value, 2) for key, value in self.timings_ms.items()},
            "model_versions": dict(self.model_versions),
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_payload(), indent=indent, ensure_ascii=False)
