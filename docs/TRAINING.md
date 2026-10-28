# Training (next phase — not implemented yet)

This repository currently ships **inference only**. The training and MLOps tracks from
[`PLAN.md`](../PLAN.md) are the next phase; this document records the agreed recipe so the work is
not re-litigated later. Do not expect `training/` modules in the tree yet.

## Scope

Fine-tune pretrained checkpoints. Training BLIP from scratch (~32 A100-days) is out of scope and
would not beat the released checkpoints at this project's compute budget.

| Track | Base | Data | Method |
|---|---|---|---|
| Captioning | `Salesforce/blip-image-captioning-base` | COCO Karpathy split (113k train / 5k val / 5k test) | LoRA on the text decoder, vision tower frozen |
| Segmentation | `nvidia/segformer-b2-finetuned-ade-512-512` | ADE20K (25k) or COCO-Stuff 164k | Decoder fine-tune, encoder frozen for epoch 1 |

## Data preparation

1. COCO images and annotations: `python scripts/download_coco.py --splits train2017 val2017 --annotations --stuff`.
2. Karpathy split: map `train2017`/`val2017` image ids to the standard 113k/5k/5k split. The split
   file is small and should be committed as `data/manifests/karpathy_*.json` (ids only).
3. Optional: COCO-Stuff `stuff_train2017.json` for 171-class semantic targets (map to ADE or VOC
   as needed).
4. Record a manifest of image ids + SHA-256 so runs are reproducible without the images.

## Captioning (LoRA)

```yaml
# configs/train/blip_lora.yaml (planned)
base_model: Salesforce/blip-image-captioning-base
method: lora
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [query, key, value, dense]   # text decoder only
train:
  epochs: 1
  batch_size: 16
  grad_accum: 4
  lr: 1.0e-4
  scheduler: cosine
  warmup_ratio: 0.05
  precision: bf16
  image_size: 384
eval:
  every_steps: 1000
  metric: cider
```

Start on a 30k-image subset to validate the loop, then run the full 113k split. Save adapters (a
few MB) to the Hub instead of full checkpoints.

## Segmentation (SegFormer decoder)

- 512×512 crops, batch 16 on A100 or 4 with gradient accumulation on T4.
- Freeze the encoder for the first epoch, then unfreeze at `lr / 10`.
- 40k steps, poly schedule, mIoU on the ADE validation split every 2k steps.
- Promotion gate: must beat the pretrained checkpoint's mIoU on the held-out split; otherwise it
  stays an option and the negative result is reported.

## Tracking and artifacts

- W&B project `vision-ai-studio`; log git SHA, config hash, manifest hash, metrics, sample images.
- Push adapters/checkpoints to HF Hub with a model card containing dataset, metrics, license and
  the exact revision.
- Record the pulled revision and adapter repo id in the deployment environment
  (`VAS_CAPTION_REVISION`, `VAS_LORA_REPO`); the app reports the revision it loaded in
  `AnalysisResult.model_versions`.

## Compute

GPU work runs on Kaggle (free T4/P100) or Colab, never on the Space. A full LoRA epoch on 113k
images at micro-batch 16 on a T4 is on the order of 20–40 hours; validate on 30k first.
Estimates must be replaced with measured wall-clock times in the run report.

## Definition of done

- One fine-tuned captioner beats BLIP base on CIDEr on a fixed 1k-image manifest, or the negative
  result is documented with numbers.
- Both training runs reproduce from config + git SHA + manifest.
- The app can load the adapter via configuration and reports its revision in
  `AnalysisResult.model_versions`.
