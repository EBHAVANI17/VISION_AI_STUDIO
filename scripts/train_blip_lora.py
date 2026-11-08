#!/usr/bin/env python
"""CLI runner for BLIP LoRA fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PIL import Image

from vision_ai_studio.training.blip_lora import BlipLoraTrainingConfig, train_blip_lora
from vision_ai_studio.tracking import get_tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BLIP with LoRA")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="artifacts/models/blip-lora")
    parser.add_argument("--tracking", type=str, default="none", choices=["mlflow", "wandb", "none"])
    parser.add_argument("--synthetic", action="store_true", help="Run quick synthetic test training")
    args = parser.parse_args()

    cfg = BlipLoraTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )
    tracker = get_tracker(system=args.tracking, experiment_name="vision-ai-studio-blip-lora")

    if args.synthetic:
        print("Generating 10 synthetic samples for quick test run...")
        samples = [
            (Image.new("RGB", (384, 384), color=(i * 20, 100, 150)), f"a synthetic test scene {i}")
            for i in range(10)
        ]
    else:
        # Load from fixtures or data
        sample_img = Path("tests/fixtures/sample.jpg")
        if not sample_img.exists():
            print("Error: tests/fixtures/sample.jpg not found and no dataset provided.")
            sys.exit(1)
        samples = [(sample_img, "a room with a person and furniture")] * 16

    print(f"Starting LoRA training with {len(samples)} samples on device: {cfg.device}")
    output_dir = train_blip_lora(samples, cfg=cfg, tracker=tracker)
    print(f"LoRA fine-tuning finished! Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
