# Evaluation

The harness measures each model on the dataset its weights were trained for. Numbers are only
comparable when the image manifest, resize edge and class mapping are identical, so every report
records those fields.

## Install extras

```bash
pip install -e ".[eval]"
```

`pycocoevalcap` needs a Java runtime for METEOR and SPICE. Everything else is pure Python.
The ADE20K download requires accepting the dataset terms at
<https://groups.csail.mit.edu/vision/datasets/ADE20K/>; the scripts do not bypass that.

## Data

```bash
python scripts/download_coco.py --dest data/coco --splits val2017 --annotations
# optional: train2017 images and COCO-Stuff annotations
python scripts/download_coco.py --dest data/coco --splits train2017 --annotations --stuff
```

Resulting layout:

```
data/coco/
  val2017/*.jpg
  annotations/captions_val2017.json
  annotations/instances_val2017.json
  annotations/stuff_train2017.json      # optional
```

ADE20K is expected as:

```
data/ADEChallengeData2016/
  images/validation/ADE_val_00000001.jpg
  annotations/validation/ADE_val_00000001.png
```

## Segmentation

```bash
# DeepLabV3 COCO/VOC on COCO val2017 (500 images by default)
python scripts/evaluate_segmentation.py --mode coco-voc --seg voc --coco-root data/coco --limit 500

# SegFormer ADE20K on the ADE validation split
python scripts/evaluate_segmentation.py --mode ade --seg ade \
    --images data/ADEChallengeData2016/images/validation \
    --masks  data/ADEChallengeData2016/annotations/validation
```

Reports are written to `artifacts/eval/segmentation_<mode>.json` and printed as tables.

### What is measured

- **IoU** per class: `TP / (TP + FP + FN)` over all evaluated pixels.
- **mIoU (all present classes)**: mean over classes that occur in the ground truth, background
  included.
- **mIoU (foreground)**: same, excluding class 0.
- **Pixel accuracy**: diagonal / total.

### COCO/VOC conversion caveats

- COCO has 80 thing categories; only 20 map to VOC. `banana`, `laptop`, `sports ball`, etc. become
  **background**, exactly as in the 21-class training target.
- Crowd regions are marked ignore, not background.
- Overlapping instances are painted large-to-small, so the smaller object wins where masks overlap.
- Instance polygons are rasterized with `pycocotools.annToMask`; this is not identical to a
  hand-labelled semantic mask, but it is deterministic and documented.

### ADE caveats

- Label PNGs use 0 for ignore and 1..150 for classes. The HF SegFormer head uses 0..149, so the
  harness maps `raw - 1` and ignores raw values of 0 or >150.
- `--max-edge` (default 640) controls resize. Do not compare runs with different values.

## Captioning

```bash
python scripts/evaluate_captioning.py \
    --coco-root data/coco --caption blip-base --limit 500 --beam 1 --max-edge 1024
```

Outputs:

- `artifacts/eval/captioning_<variant>.json` — scores, model version, caveats;
- `artifacts/eval/captioning_<variant>_predictions.json` — raw captions, so scores can be
  recomputed without the model.

Metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr; add `--spice` for SPICE. Each metric is wrapped
individually: a missing Java runtime records an error for that metric instead of aborting the run.
References are the five COCO captions per image.

## Benchmark

```bash
python scripts/benchmark.py --image tests/fixtures/sample.jpg --sizes 512 1024 2048 --runs 3
```

Reports total p50/p95 and per-stage medians, after a warm-up run that excludes model loading.
Use the same machine and thread count when comparing commits.

## Reproducibility

- Sampling uses `--seed` over sorted image ids; report the seed.
- Reports include the model version string (HF commit hash when available, or the torchvision
  weight enum).
- Pin captioner/segmenter revisions with `VAS_CAPTION_REVISION` / `VAS_SEGMENTER_REVISION` for
  published numbers.
- Keep `artifacts/eval/*.json` for the runs referenced in [`RESULTS.md`](RESULTS.md).

## Cost

Rough CPU times per 1 000 images at `--max-edge 640`:

| Step | Estimate |
|---|---|
| DeepLabV3-R50 inference | 25–45 min |
| SegFormer-B2 inference | 20–35 min |
| BLIP-base captioning | 30–60 min |
| Scoring (caption metrics) | 1–3 min |

Use `--limit 100` while iterating.
