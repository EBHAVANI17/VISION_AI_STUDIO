"""BLIP LoRA fine-tuning for image captioning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vision_ai_studio.tracking import ExperimentTracker, NullTracker

logger = logging.getLogger(__name__)


@dataclass
class BlipLoraTrainingConfig:
    base_model: str = "Salesforce/blip-image-captioning-base"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("query", "key", "value", "dense")
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "artifacts/models/blip-lora"


class CaptionDataset(Dataset):
    """Simple in-memory or path-based caption dataset."""

    def __init__(self, samples: list[tuple[str | Path | Image.Image, str]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str]:
        item, caption = self.samples[idx]
        if isinstance(item, (str, Path)):
            image = Image.open(item).convert("RGB")
        else:
            image = item.convert("RGB")
        return image, caption


def collate_fn_builder(processor: Any, max_length: int = 40) -> Callable:
    def collate_fn(batch: list[tuple[Image.Image, str]]) -> dict[str, torch.Tensor]:
        images = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        encoding = processor(images=images, text=texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        # BLIP uses input_ids as labels for conditional generation
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding

    return collate_fn


def setup_lora_blip(base_model_name: str, r: int, alpha: int, dropout: float, target_modules: tuple[str, ...]) -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(base_model_name)
    model = BlipForConditionalGeneration.from_pretrained(base_model_name)

    # Freeze vision model to only adapt text decoder
    for param in model.vision_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
    return peft_model, processor


def train_blip_lora(
    train_samples: list[tuple[str | Path | Image.Image, str]],
    val_samples: list[tuple[str | Path | Image.Image, str]] | None = None,
    cfg: BlipLoraTrainingConfig | None = None,
    tracker: ExperimentTracker | None = None,
) -> Path:
    """Train or fine-tune BLIP with LoRA on captioning dataset."""
    cfg = cfg or BlipLoraTrainingConfig()
    tracker = tracker or NullTracker()
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tracker.log_params(
        {
            "base_model": cfg.base_model,
            "r": cfg.r,
            "alpha": cfg.alpha,
            "dropout": cfg.dropout,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "train_samples": len(train_samples),
        }
    )

    model, processor = setup_lora_blip(
        base_model_name=cfg.base_model,
        r=cfg.r,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
        target_modules=cfg.target_modules,
    )

    device = torch.device(cfg.device)
    model.to(device)

    train_dataset = CaptionDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_builder(processor),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    global_step = 0

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            loss_val = loss.item() * cfg.gradient_accumulation_steps
            epoch_loss += loss_val

            if step % 10 == 0:
                tracker.log_metrics({"train_loss": loss_val}, step=global_step)
                logger.info("Epoch %d Step %d Loss: %.4f", epoch + 1, step, loss_val)

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        tracker.log_metrics({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1}, step=global_step)
        logger.info("Epoch %d Complete - Avg Loss: %.4f", epoch + 1, avg_epoch_loss)

    # Save fine-tuned adapter weights
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))
    tracker.log_artifact(str(output_path / "adapter_model.safetensors"))
    tracker.finish()

    logger.info("Saved LoRA adapter to %s", output_path)
    return output_path
