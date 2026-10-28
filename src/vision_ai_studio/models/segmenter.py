"""Semantic segmentation backends.

Both backends are imported lazily so that the package (and its unit tests) can be used
without PyTorch installed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vision_ai_studio.models.base import (
    ModelInferenceError,
    ModelLoadError,
    Segmentation,
)

ADE_REPO = "nvidia/segformer-b2-finetuned-ade-512-512"
MAX_INFERENCE_EDGE = 640  # keeps CPU inference in the low seconds


class DeepLabV3VOCSegmenter:
    """torchvision DeepLabV3-ResNet50 trained on COCO with the 21 VOC labels."""

    name = "deeplabv3-voc21"

    def __init__(self, device: str = "cpu", revision: str | None = None) -> None:
        del revision  # torchvision weights are versioned by torchvision itself
        try:
            import torch
            from torchvision.models.segmentation import (
                DeepLabV3_ResNet50_Weights,
                deeplabv3_resnet50,
            )
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ModelLoadError(
                "torchvision is required for the VOC segmenter. Install with `pip install -e .`."
            ) from exc

        self._torch = torch
        self._device = torch.device(device)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.labels = tuple(_display_label(str(label)) for label in weights.meta["categories"])
        self.version = f"torchvision:deeplabv3_resnet50:{weights.name}"
        self._model = deeplabv3_resnet50(weights=weights).to(self._device).eval()

    def predict(self, image: Image.Image) -> Segmentation:
        torch = self._torch
        from torchvision.transforms import functional as tvf

        try:
            original_size = image.size
            prepared = _shrink(image, MAX_INFERENCE_EDGE)
            tensor = tvf.to_tensor(prepared)
            tensor = tvf.normalize(
                tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ).unsqueeze(0)

            with torch.inference_mode():
                logits = self._model(tensor.to(self._device))["out"]
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=(original_size[1], original_size[0]),
                    mode="bilinear",
                    align_corners=False,
                )
                mask = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.int32)
        except Exception as exc:
            raise ModelInferenceError(f"DeepLabV3 inference failed: {exc}") from exc

        return Segmentation(mask=mask, labels=self.labels, version=self.version)


class SegFormerADESegmenter:
    """Hugging Face SegFormer-B2 fine-tuned on ADE20K (150 classes)."""

    name = "segformer-ade150"

    def __init__(
        self, device: str = "cpu", revision: str | None = None, repo_id: str = ADE_REPO
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ModelLoadError(
                "transformers and torch are required for the ADE20K segmenter."
            ) from exc

        self._torch = torch
        self._device = torch.device(device)
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                repo_id, revision=revision or "main"
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                repo_id, revision=revision or "main"
            )
            self._model = model.to(self._device).eval()  # type: ignore[arg-type]  # transformers stubs mistype Module.to
        except Exception as exc:
            raise ModelLoadError(f"Could not load {repo_id}: {exc}") from exc

        label_map = {int(key): str(value) for key, value in self._model.config.id2label.items()}
        count = int(self._model.config.num_labels)
        self.labels = tuple(label_map.get(index, f"class {index}") for index in range(count))
        commit = getattr(self._model.config, "_commit_hash", None) or revision or "main"
        self.version = f"hf:{repo_id}@{commit}"

    def predict(self, image: Image.Image) -> Segmentation:
        torch = self._torch
        try:
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = inputs.to(self._device)
            with torch.inference_mode():
                logits = self._model(**inputs).logits
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            mask = upsampled.argmax(dim=1)[0].to("cpu").numpy().astype(np.int32)
        except Exception as exc:
            raise ModelInferenceError(f"SegFormer inference failed: {exc}") from exc

        return Segmentation(mask=mask, labels=self.labels, version=self.version)


def build_segmenter(
    name: str, *, device: str = "cpu", revision: str | None = None
) -> DeepLabV3VOCSegmenter | SegFormerADESegmenter:
    if name == "voc":
        return DeepLabV3VOCSegmenter(device=device, revision=revision)
    if name == "ade":
        return SegFormerADESegmenter(device=device, revision=revision)
    raise ValueError(f"Unknown segmenter {name!r}")


def _display_label(label: str) -> str:
    """torchvision VOC weights use ``__background__``; normalize pseudo-classes."""
    cleaned = label.strip().strip("_")
    return cleaned or label


def _shrink(image: Image.Image, max_edge: int) -> Image.Image:
    longest = max(image.size)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)
