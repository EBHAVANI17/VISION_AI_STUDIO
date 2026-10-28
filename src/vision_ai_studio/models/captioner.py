"""Image captioning with BLIP (optionally with a PEFT/LoRA adapter)."""

from __future__ import annotations

from PIL import Image

from vision_ai_studio.models.base import (
    ModelInferenceError,
    ModelLoadError,
)

BLIP_REPOS: dict[str, str] = {
    "blip-base": "Salesforce/blip-image-captioning-base",
    "blip-large": "Salesforce/blip-image-captioning-large",
}


class BlipCaptioner:
    name = "blip"

    def __init__(
        self,
        variant: str = "blip-base",
        *,
        device: str = "cpu",
        revision: str | None = None,
        lora_repo: str | None = None,
        lora_revision: str | None = None,
        max_length: int = 40,
    ) -> None:
        if variant not in BLIP_REPOS:
            raise ValueError(f"Unknown BLIP variant {variant!r}")
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ModelLoadError("transformers and torch are required for captioning.") from exc

        self._torch = torch
        self._device = torch.device(device)
        self._repo = BLIP_REPOS[variant]
        self.max_length = max_length

        try:
            self._processor = BlipProcessor.from_pretrained(self._repo, revision=revision or "main")
            model = BlipForConditionalGeneration.from_pretrained(
                self._repo, revision=revision or "main"
            )
            self._model = model.to(self._device)  # type: ignore[arg-type]  # transformers stubs mistype Module.to
        except Exception as exc:
            raise ModelLoadError(f"Could not load {self._repo}: {exc}") from exc

        commit = getattr(self._model.config, "_commit_hash", None) or revision or "main"
        version = f"hf:{self._repo}@{commit}"
        if lora_repo:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ModelLoadError(
                    "Install `vision-ai-studio[lora]` to load a LoRA adapter."
                ) from exc
            try:
                self._model = PeftModel.from_pretrained(
                    self._model, lora_repo, revision=lora_revision
                ).to(self._device)
            except Exception as exc:
                raise ModelLoadError(f"Could not load LoRA adapter {lora_repo}: {exc}") from exc
            version += f"+lora:{lora_repo}@{lora_revision or 'main'}"

        self._model.eval()
        self.version = version

    def caption(self, image: Image.Image, *, beam: int = 1) -> str:
        torch = self._torch
        try:
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    num_beams=max(1, int(beam)),
                    max_length=self.max_length,
                )
            text = self._processor.decode(generated[0], skip_special_tokens=True)
        except Exception as exc:
            raise ModelInferenceError(f"BLIP captioning failed: {exc}") from exc
        return str(text).strip()


def build_captioner(
    variant: str,
    *,
    device: str = "cpu",
    revision: str | None = None,
    lora_repo: str | None = None,
) -> BlipCaptioner:
    return BlipCaptioner(
        variant,
        device=device,
        revision=revision,
        lora_repo=lora_repo,
    )
