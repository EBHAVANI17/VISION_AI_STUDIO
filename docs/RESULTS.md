# Results

This file is intentionally **pending**. The evaluation harness is implemented and documented in
[`EVALUATION.md`](EVALUATION.md); values are only added here after they are produced by the
scripts, so there are no invented or stale numbers.

## Segmentation

| Model | Dataset | Images | max-edge | mIoU (all) | mIoU (foreground) | Pixel accuracy |
|---|---|---|---|---|---|---|
| DeepLabV3-R50 (COCO/VOC-21) | COCO val2017 | pending | 640 | pending | pending | pending |
| SegFormer-B2 (ADE20K-150) | ADE20K val | pending | 640 | pending | pending | pending |

## Captioning

| Model | Images | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| BLIP base | pending | pending | pending | pending | pending |
| BLIP large | pending | pending | pending | pending | pending |

## Latency (CPU)

| Model | Input | p50 total | p95 total | Caption | Segment |
|---|---|---|---|---|---|
| BLIP base + SegFormer-B2 | pending | pending | pending | pending | pending |

## Reproduce

```bash
python scripts/download_coco.py --dest data/coco --splits val2017 --annotations
python scripts/evaluate_segmentation.py --mode coco-voc --seg voc --limit 500
python scripts/evaluate_captioning.py --caption blip-base --limit 500
python scripts/benchmark.py --sizes 512 1024 2048 --runs 3
```

Record the machine, thread count, `--max-edge`, seed and model revision next to any published
number. The JSON reports under `artifacts/eval/` contain all of them except the machine.
