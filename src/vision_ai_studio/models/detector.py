"""Object detection with YOLO (Ultralytics) and test fakes."""

from __future__ import annotations

import logging
from typing import Any
from PIL import Image

from vision_ai_studio.models.base import ModelInferenceError, ModelLoadError
from vision_ai_studio.schemas import DetectedObject

logger = logging.getLogger(__name__)


class FakeDetector:
    """Deterministic fake detector for testing."""

    name = "fake-yolo"
    version = "fake:yolo@v1"

    def detect(self, image: Image.Image, *, threshold: float = 0.25) -> list[DetectedObject]:
        w, h = image.size
        # Deterministic fake boxes
        return [
            DetectedObject(
                label="person",
                confidence=0.92,
                box=(0.1 * w, 0.1 * h, 0.6 * w, 0.9 * h),
            ),
            DetectedObject(
                label="chair",
                confidence=0.85,
                box=(0.6 * w, 0.4 * h, 0.9 * w, 0.85 * h),
            ),
        ]


class YoloDetector:
    """YOLO object detector backed by Ultralytics."""

    name = "yolo"

    def __init__(self, model_name: str = "yolov8n.pt", *, device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError("ultralytics is required for YOLO detection.") from exc

        self.model_name = model_name
        self.device = device

        try:
            self._model = YOLO(model_name)
        except Exception as exc:
            raise ModelLoadError(f"Could not load YOLO model {model_name}: {exc}") from exc

        self.version = f"ultralytics:{model_name}"

    def detect(self, image: Image.Image, *, threshold: float = 0.25) -> list[DetectedObject]:
        try:
            results = self._model(image, conf=threshold, device=self.device, verbose=False)
            detections: list[DetectedObject] = []
            if not results:
                return detections

            r = results[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                return detections

            names = r.names
            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                detections.append(
                    DetectedObject(
                        label=label,
                        confidence=conf,
                        box=(coords[0], coords[1], coords[2], coords[3]),
                    )
                )
            return detections
        except Exception as exc:
            raise ModelInferenceError(f"YOLO detection failed: {exc}") from exc


def build_detector(variant: str = "yolov8n.pt", *, device: str = "cpu", fake: bool = False) -> Any:
    if fake or variant == "fake":
        return FakeDetector()
    return YoloDetector(variant, device=device)
