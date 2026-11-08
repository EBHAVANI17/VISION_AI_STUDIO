from __future__ import annotations

import io
from fastapi.testclient import TestClient
from PIL import Image

from vision_ai_studio.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_info_endpoint() -> None:
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "available_segmenters" in data
    assert "available_captioners" in data
    assert "supported_languages" in data


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "vas_requests_total" in text


def test_analyze_endpoint_fake(monkeypatch) -> None:
    monkeypatch.setenv("VAS_MODELS", "fake")
    img = Image.new("RGB", (200, 200), color=(120, 80, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/analyze",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        params={"segmenter": "ade", "captioner": "blip-base", "enable_detector": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "caption" in payload
    assert "segments" in payload
    assert "detections" in payload
    assert len(payload["detections"]) > 0
    assert payload["detections"][0]["label"] == "person"
