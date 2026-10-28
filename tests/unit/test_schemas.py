from __future__ import annotations

import json

from vision_ai_studio.schemas import AnalysisResult, Segment


def make_result() -> AnalysisResult:
    return AnalysisResult(
        caption="a red box and a green circle",
        segments=(Segment(class_id=1, label="person", pixels=1200, coverage=0.6),),
        dominant_label="person",
        image_size=(320, 240),
        overlay_png=b"\x89PNG-overlay",
        mask_png=b"\x89PNG-mask",
        translated="एक लाल बॉक्स और एक हरा सर्कल",
        target_lang="hi",
        timings_ms={"caption": 12.3456, "segment": 99.0},
        model_versions={"captioner": "fake:captioner@1"},
        warnings=("a warning",),
    )


def test_payload_is_json_safe() -> None:
    payload = make_result().to_payload()
    text = json.dumps(payload)
    assert "PNG" not in text
    assert payload["segments"][0]["coverage"] == 0.6
    assert payload["timings_ms"]["caption"] == 12.35
    assert payload["image_size"] == {"width": 320, "height": 240}
    assert payload["warnings"] == ["a warning"]


def test_to_json_roundtrip() -> None:
    text = make_result().to_json()
    payload = json.loads(text)
    assert payload["caption"].startswith("a red box")
    assert payload["translated"]


def test_dominant_segment_property() -> None:
    result = make_result()
    assert result.dominant_segment is not None
    assert result.dominant_segment.label == "person"
