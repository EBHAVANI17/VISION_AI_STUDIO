from __future__ import annotations

from PIL import Image

from vision_ai_studio.models.detector import FakeDetector, build_detector


def test_fake_detector() -> None:
    detector = FakeDetector()
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    detections = detector.detect(img)
    assert len(detections) == 2
    assert detections[0].label == "person"
    assert detections[0].confidence > 0.8
    assert len(detections[0].box) == 4


def test_build_detector_fake() -> None:
    detector = build_detector("fake", fake=True)
    assert isinstance(detector, FakeDetector)
