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
References are the five COCO captions per image. Install a JRE (for example
`apt-get install default-jre` in the container) to enable METEOR and SPICE.

## Error analysis

The evaluation reports contain the full confusion matrix and per-image rows, so the failure
analysis does not need to re-run a model:

```bash
python scripts/analyze_results.py --limit 10
```

This writes `artifacts/eval/analysis.md` with, per segmentation track:

- weakest and strongest classes (classes with no ground-truth pixels are excluded, so a 0.000 IoU
  always means "present but always wrong");
- the most frequent truth→prediction confusions with the share of the truth class they consume;
- over- and under-segmentation ratios (predicted pixels / ground-truth pixels);
- the worst images by per-image foreground mIoU.

For captioning it adds the duplicate rate, caption-length distribution, a token-F1 ranking against
the best reference, and the weakest captions. Token F1 is a triage signal for finding failure cases
— BLEU and CIDEr remain the reported metrics. The captions are saved separately from the scores, so
the analysis can be re-run without loading the model.

Recommended loop for a change (new checkpoint, resize, decoding):

1. run the 300-image evaluation with the same seed;
2. run the analysis and compare the same tables, not just the headline mIoU;
3. only then spend time on the full validation split.

## Benchmark

```bash
make benchmark
# equivalent:
python scripts/benchmark.py --image tests/fixtures/sample.jpg \
    --seg ade --caption blip-base --sizes 512 1024 2048 --runs 3
```

Reports total p50/p95 and per-stage medians, after a warm-up run that excludes model loading.
Use the same machine and thread count when comparing commits. Note that both models resize
internally (SegFormer to 512 px, BLIP to 384 px), so larger inputs mostly add preprocessing,
logit-upsampling and rendering cost rather than model compute.

## Convenience targets

```bash
make eval-seg      # both segmenters, 300 images, seed 0, max-edge 640
make eval-caption  # BLIP base, 300 images, beam 1
make eval-report   # artifacts/eval/analysis.md
make benchmark     # latency table above
```

Override the sample size with `make eval-seg EVAL_N=1000` (and `make eval-caption EVAL_N=1000`).
`VAS_SEGMENTER_REVISION` / `VAS_CAPTION_REVISION` pin the HF revisions if `main` moves.

## Reproducibility

- Sampling uses `--seed` over sorted image ids; report the seed.
- Reports include the model version string (HF commit hash when available, or the torchvision
  weight enum).
- Pin captioner/segmenter revisions with `VAS_CAPTION_REVISION` / `VAS_SEGMENTER_REVISION` for
  published numbers.
- Keep `artifacts/eval/*.json` for the runs referenced in [`RESULTS.md`](RESULTS.md).

## Cost

Measured on the results machine (4 inference threads, desktop load ~70 %), `--max-edge 640`:

| Step | Observation |
|---|---|
| DeepLabV3-R50 on 300 COCO images | ~20 min (~4 s/image) |
| SegFormer-B2 on 300 ADE images | ~23 min (~4.5 s/image) |
| BLIP-base on 300 images + scoring | ~9 min (~1.7 s/image), scoring is seconds |
| Error analysis (no model) | seconds |

At those rates a 1 000-image segmentation run is roughly 65–75 min and a 1 000-image captioning
run roughly 30 min. Use `--limit 100` while iterating; the run time scales linearly with the
sample. Full per-image numbers are in [`RESULTS.md`](RESULTS.md).
