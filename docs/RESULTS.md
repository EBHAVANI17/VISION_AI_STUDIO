# Results

Measured on 2026-11-01 on CPU. Every number below comes from the scripts in this repository and
the raw reports are kept under `artifacts/eval/` (git-ignored; regenerate with `make eval-seg`,
`make eval-caption`, and `make benchmark`). Nothing here is copied from a model card.

## Headline

| Task | Model | Dataset (sample) | Headline metric | Secondary |
|---|---|---|---|---|
| Segmentation | DeepLabV3-R50 (COCO/VOC-21) | COCO val2017, 300 images | **foreground mIoU 0.5947** | mIoU all 0.6103, pixel acc 0.9269 |
| Segmentation | SegFormer-B2 (ADE20K-150) | ADE20K val, 300 images | **mIoU 0.4480** | foreground mIoU 0.4457, pixel acc 0.8150 |
| Captioning | BLIP base | COCO val2017, 300 images | **CIDEr 0.8266** | BLEU-4 0.2555, ROUGE-L 0.4375 |
| Latency (CPU) | BLIP base + SegFormer-B2 | 512→2048 px | **p50 6.5–9.7 s** | p95 6.7–9.8 s |

METEOR and SPICE are absent because the evaluation machine has no Java runtime; the harness records
that as a per-metric error instead of failing the run. Translation and speech quality were not
measured (see [What is not measured](#what-is-not-measured)).

## Setup

| | |
|---|---|
| CPU | 13th Gen Intel Core i5-13420H, 8 cores / 12 threads |
| RAM | 15.6 GB, shared with a desktop IDE and browser (load ~70–75 % during runs) |
| OS | Windows 11 Home 10.0.26300 |
| Inference threads | 4 (`registry._configure_threads` caps `torch.set_num_threads`) |
| Python | 3.13.5 |
| PyTorch / torchvision | 2.14.1+cpu / 0.29.1+cpu |
| Transformers / NumPy / Pillow | 5.17.0 / 2.5.2 / 12.3.0 |
| pycocotools / pycocoevalcap | 2.0.11 / 1.2 |
| Device | CPU only; no GPU numbers |

Pinned checkpoints (reported verbatim in each JSON report):

| Component | Version string |
|---|---|
| DeepLabV3-R50 | `torchvision:deeplabv3_resnet50:COCO_WITH_VOC_LABELS_V1` |
| SegFormer-B2 | `hf:nvidia/segformer-b2-finetuned-ade-512-512@de01bae28967510f9ddd496c60a969357195400c` |
| BLIP base | `hf:Salesforce/blip-image-captioning-base@82a37760796d32b1411fe092ab5d4e227313294b` |

## Protocol

- **Sampling.** 300 images drawn with `random.Random(0).sample` over sorted image ids: 300/5 000
  COCO val2017 images and 300/2 000 ADE20K validation images. The same seed reproduces the same
  list; the reports store per-image rows and the full confusion matrix.
- **Resize.** Segmentation `--max-edge 640`. At this setting both backends run at their internal
  cap — DeepLab shrinks to 640 px (`segmenter.MAX_INFERENCE_EDGE`) and SegFormer's processor
  resizes to 512×512 — and the prediction is upsampled back to the loaded image size. Ground-truth
  masks are resized with nearest-neighbour to the prediction resolution. Captioning source at
  `--max-edge 1024`, beam 1. BLIP's processor then resizes to the model input (384 px), so caption
  scores are not a 1024-px-input result.
- **Metric definitions.** IoU = TP / (TP + FP + FN) over aggregated pixels; mIoU averages over
  classes present in the ground truth (`background` included). Foreground mIoU excludes
  `background`. Pixel accuracy = diagonal / all pixels.
- **Confidence.** The 95 % intervals below are normal approximations over per-image foreground mIoU
  (VOC n = 227, ADE n = 300). Corpus-level caption metrics (BLEU/CIDEr) do not get an interval.
- **No test-set tuning.** These models were not fine-tuned; the evaluation measures frozen
  checkpoints, so the numbers describe the shipped pipeline rather than a trained result.

## Segmentation: COCO/VOC-21

Ground truth is built from COCO instance polygons mapped onto 20 VOC foreground classes; COCO
categories without a VOC equivalent (`fire hydrant`, `bench`, `spoon`, `traffic light`, …) become
background, exactly as in the training target. Crowd regions are ignored.

| Metric | Value |
|---|---|
| Images | 300 (0 skipped) |
| mIoU (all 21 present classes) | 0.6103 |
| Foreground mIoU (20 classes) | 0.5947 |
| Pixel accuracy | 0.9269 |
| Per-image fg mIoU (227 scored) | 0.6107, 95 % CI [0.5751, 0.6464] |
| Per-image median / p10 / p90 | 0.6945 / 0.1699 / 0.9079 |

73 of the 300 sampled images contain no VOC foreground at all (only categories that map to
background or crowd), so they are excluded from the per-image foreground statistic while still
contributing background pixels to the global metrics. Pixel accuracy is background-dominated:
0.9269 mostly says "the model finds the background", not "the objects are right".

### Per-class results

| Class | IoU | GT pixels | Predicted pixels |
|---|---|---|---|
| sofa | 0.186 | 255,963 | 607,460 |
| boat | 0.239 | 457,968 | 129,354 |
| chair | 0.305 | 599,183 | 430,751 |
| pottedplant | 0.337 | 118,866 | 323,995 |
| diningtable | 0.344 | 1,849,933 | 2,660,334 |
| bottle | 0.446 | 193,115 | 178,076 |
| tvmonitor | 0.454 | 281,697 | 227,748 |
| bicycle | 0.528 | 218,523 | 217,937 |
| bird | 0.536 | 60,244 | 40,627 |
| cow | 0.600 | 121,130 | 179,927 |
| sheep | 0.626 | 192,737 | 214,979 |
| horse | 0.700 | 158,075 | 136,606 |
| dog | 0.728 | 408,748 | 517,451 |
| cat | 0.774 | 568,677 | 653,333 |
| car | 0.792 | 572,672 | 576,140 |
| motorbike | 0.815 | 519,550 | 502,944 |
| person | 0.836 | 6,064,504 | 6,401,299 |
| aeroplane | 0.874 | 95,449 | 95,565 |
| train | 0.875 | 808,173 | 835,377 |
| bus | 0.900 | 756,702 | 748,885 |
| background | 0.922 | 67,249,620 | 65,872,741 |

All 21 classes are present in the sample, so mIoU averages over 21 rows (20 foreground + background).

### Error patterns

**Objects disappear into background.** The three weakest large classes are thin or small things
against textured backgrounds: 74.2 % of `boat` pixels and 52.9 % of `chair` pixels are predicted as
background; `boat` is predicted for only 0.28× its true area. In contrast, foreground-heavy classes
are reliable (`bus` 0.900, `train` 0.875, `aeroplane` 0.874, `person` 0.836).

**Furniture bleeds.** `sofa` is the weakest class (0.186) yet is over-predicted 2.37×, and 12 % of
`diningtable` truth is labelled `sofa`. `diningtable` itself loses 20.5 % to background and is
over-predicted 1.44×. `pottedplant` is over-predicted 2.73×. This is a coarse-texture confusion
(sofa fabric, table surfaces, plants indoors), not a detection failure.

**Largest single confusion.** `background → diningtable` accounts for 1,472,713 pixels (2.2 % of
all background); `background → person` is next at 668,500 (1.0 %). Both are conservative false
positives that matter for the UI's class ranking even when IoU stays high.

### Worst images

Nine of the ten worst images are scenes where the mapped foreground covers at most 2.1 % of the
image (247–6 485 px: a person, two birds, a boat, bottles, cars, chairs). In every one of the nine,
DeepLab predicts **100 % background**, so pixel accuracy stays between 0.979 and 0.999 while
foreground mIoU is exactly 0. The remaining ground truth is background because the other annotated
categories (`fire hydrant`, `sink`, `traffic light`, books, cups, …) have no VOC equivalent. The
metric is right to count these as foreground misses; the per-image list is what exposes that
background-heavy accuracy hides them.

The exception is `000000129113.jpg`: a close-up of a spoon on a woven bamboo placemat, annotated as
`dining table` (the placemat). DeepLab predicts **75.2 % sofa, 23.3 % chair, 1.4 % background** and
0 % dining table, giving pixel accuracy 0.0089. Reproducing that single image confirms the failure
is texture: the regular weave resembles upholstery, and the COCO label is itself coarse. This is
the kind of case a per-image worst list is for.

## Segmentation: ADE20K-150

Ground truth uses the official label PNGs (`raw - 1`, raw 0/255 ignored). 142 of the 150 classes
appear in the sample; mIoU averages over those 142.

| Metric | Value |
|---|---|
| Images | 300 (0 skipped) |
| mIoU (142 present classes) | 0.4480 |
| Foreground mIoU (141 classes) | 0.4457 |
| Pixel accuracy | 0.8150 |
| Per-image fg mIoU (300 scored) | 0.5333, 95 % CI [0.5133, 0.5533] |
| Per-image median / p10 / p90 | 0.5260 / 0.3235 / 0.7701 |
| mIoU on classes with ≥10k GT px (131) | 0.4706 |

Per-image mean (0.5333) is higher than the global mIoU (0.4480) because global IoU weights every
pixel of a class, and the large-area classes are exactly where sibling confusions concentrate.
Restricting to classes with at least 10 000 ground-truth pixels recovers 2.3 points (0.4706),
quantifying the long-tail cost.

### Per-class results

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.951 | tower | 0.000 |
| runway | 0.933 | escalator | 0.000 |
| pool table | 0.907 | buffet | 0.000 |
| refrigerator | 0.896 | barrel | 0.000 |
| washer | 0.895 | bag | 0.000 |
| bed | 0.878 | oven | 0.000 |
| stove | 0.878 | ball | 0.000 |
| tank | 0.870 | lake | 0.000 |
| toilet | 0.868 | tray | 0.000 |
| microwave | 0.848 | crt screen | 0.000 |

Eleven present classes score 0.000; the table shows ten and `shower` is the eleventh. They account
for 7.7 % of the 142 present classes, and each covers less than 65k ground-truth pixels in the
sample (`tower`, the largest, has 63.5k). The zero scores are not "object not found" in most
cases — they are systematic sibling substitutions:

| Truth | Top prediction (share of truth pixels) |
|---|---|
| tower | building 95.4 % |
| lake | river 84.2 %, water 10.4 % |
| oven | wall 100.0 % |
| bag | wall 45.3 %, person 17.7 % |
| escalator | bridge 56.2 %, stairway 14.6 % |
| crt screen | screen 44.1 %, wall 24.7 %, monitor 21.7 % |
| ball | plaything 65.4 %, wall 32.5 % |

`oven → wall` is the one likely annotation/label-noise case (a built-in oven flush with cabinetry)
rather than a taxonomy choice.

### Error patterns

The top confusions are all taxonomy boundaries, and they are large:

| Truth | Predicted | Pixels | Share of truth |
|---|---|---|---|
| ceiling | wall | 238,997 | 8.4 % |
| door | wall | 204,449 | 27.0 % |
| water | sea | 190,179 | 24.9 % |
| field | grass | 189,436 | 63.0 % |
| mountain | earth | 159,233 | 12.6 % |
| rug | floor | 156,155 | 52.6 % |
| wall | ceiling | 147,419 | 1.5 % |
| tree | plant | 147,011 | 5.4 % |
| road | sidewalk | 134,351 | 5.8 % |
| tree | building | 130,140 | 4.8 % |

Over-segmentation is concentrated in large structural classes (`grandstand` 3.67×, `river` 3.09×,
`bridge` 2.40×); under-segmentation hits rare indoor objects (`ball` 0.00×, `buffet` 0.01×,
`blanket` 0.01×, `fountain` 0.02×, `oven` 0.03×). The model is confident about the scene layout and
weak about the fine-grained label the benchmark asks for.

### Worst images

`ADE_val_00000923.jpg` has two classes and foreground mIoU 0 (same tiny-object pattern as VOC).
`ADE_val_00001069.jpg` is the opposite: 8 classes, pixel accuracy 0.1761, foreground mIoU 0.0366 —
a genuine scene-level failure rather than a tiny-object miss. The other eight worst images sit
between 0.13 and 0.23, mostly cluttered indoor or mixed outdoor scenes where the sibling
confusions above compound.

## Captioning: COCO val2017 (BLIP base)

| Metric | Value |
|---|---|
| Images | 300 (0 skipped) |
| BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4 | 0.6010 / 0.4610 / 0.3422 / 0.2555 |
| ROUGE-L | 0.4375 |
| CIDEr | 0.8266 |
| METEOR | skipped — no Java runtime (`WinError 2`) |
| SPICE | not requested (also needs Java) |

These are corpus-level scores against the five COCO references per image, at beam 1. They are not
directly comparable to model-card numbers measured on the full 5 000-image set with beam search
(typically beam 3), so treat them as the shipped configuration's baseline rather than a leaderboard
entry.

### Prediction patterns

| Signal | Value |
|---|---|
| Unique captions | 291 / 300 (duplicate rate 3.0 %) |
| Caption length | mean 7.1 words (min 3, max 13) |
| Token F1 vs. best reference | 0.5413, 95 % CI [0.5243, 0.5583] |
| Token F1 p10 / median / p90 | 0.333 / 0.533 / 0.750 |

Token F1 is a cheap diagnostic (set overlap with the best single reference), not a replacement for
BLEU/CIDEr; it is only used here to rank failures.

**Failure modes visible in the worst ten:**

- **Word fragments / repetition.** `a gi is eating`, `a gife eating grass`, `a row of ur urs`,
  `a group of gis eating leaves from a tree`. At beam 1 the decoder occasionally emits
  sub-word/BPE fragments for `giraffe`. This is decoding, not perception.
- **Generic or category-blurred captions.** `a clear blue sky` for a beach scene full of people and
  kites is true but omits every subject the references mention. `a bunch of cell phones on a wall`
  for a PDA, compact camera and phone is not a hallucination, but it blurs three categories into
  one. Both score near the bottom on token overlap without being factually false.
- **Over-specific captions with a wrong verb.** `a man in a black jacket and red pants snowboarding`
  describes the clothing and pose correctly, but the image is a **skier**. Adding detail did not
  reduce the metric error, because the noun/verb choice is wrong.

The 3 % duplicate rate is low for beam 1 and short captions (`a train on the tracks` three times),
so mode collapse is not the dominant error; the dominant error is lexical (fragments, wrong noun).
Beam search is the obvious next experiment.

## Latency (CPU)

`scripts/benchmark.py --seg ade --caption blip-base --sizes 512 1024 2048 --runs 3`, after one
warm-up run per size. Input is `tests/fixtures/sample.jpg` (native 1600×1067) resized to the longest
edge; 2048 therefore includes an upscale.

| Input | p50 total | p95 total | Caption p50 | Segment p50 | Render p50 |
|---|---|---|---|---|---|
| 512×341 | 6 519 ms | 6 701 ms | 2 258 ms | 4 109 ms | 47 ms |
| 1024×683 | 6 893 ms | 7 756 ms | 2 290 ms | 4 426 ms | 159 ms |
| 2048×1366 | 9 699 ms | 9 801 ms | 2 387 ms | 6 489 ms | 731 ms |

Reading the table:

- Captioning is almost flat (2.26 → 2.39 s) because BLIP's processor resizes to 384 px regardless
  of the source size. Added pixels do not buy caption detail.
- Segmentation input is also fixed — SegFormer's processor resizes to 512×512; DeepLab shrinks to
  640 px. The growth from 4.1 s to 6.5 s comes from preprocessing and from upsampling the 150-class
  logits back to 2048×1366 (a 150×1366×2048 fp32 tensor is ≈ 1.7 GB) plus the full-resolution mask
  conversion, not from a larger model input.
- Rendering is near-free at typical sizes and only becomes visible at 2048.
- This machine was shared with a desktop IDE and browser at ~70–75 % CPU load, so the absolute
  numbers are an upper bound. Compare machines only with the same thread count and idle load.

## What is not measured

Keeping the scope honest:

- **Translation** (NLLB-200 vs Google) — no BLEU/chrF or human check yet.
- **Speech** (gTTS) — no WER/MOS or audio-quality check.
- **BLIP large and the LoRA path** — supported by the code, not yet benchmarked.
- **METEOR/SPICE** — need a Java runtime. Install one and re-run `evaluate_captioning.py` with
  `--spice`; the harness will pick them up without code changes.
- **Full validation sets** — 300 images per task, not 5 000/2 000. Small differences between
  models should not be read as significant.
- **GPU latency** — CPU only.
- **Training** — no fine-tuning; every number is a frozen public checkpoint.

## Reproduce

```bash
# data (COCO is public; ADE20K requires accepting the dataset terms)
python scripts/download_coco.py --dest data/coco --splits val2017 --annotations

# segmentation (300 images, seed 0, max-edge 640)
make eval-seg
# equivalent:
python scripts/evaluate_segmentation.py --mode coco-voc --seg voc --coco-root data/coco \
    --limit 300 --seed 0 --max-edge 640
python scripts/evaluate_segmentation.py --mode ade --seg ade \
    --images data/ADEChallengeData2016/images/validation \
    --masks  data/ADEChallengeData2016/annotations/validation \
    --limit 300 --seed 0 --max-edge 640

# captioning (300 images, beam 1, source max-edge 1024)
make eval-caption
# equivalent:
python scripts/evaluate_captioning.py --coco-root data/coco --caption blip-base \
    --limit 300 --seed 0 --beam 1 --max-edge 1024

# error analysis (writes artifacts/eval/analysis.md)
make eval-report

# latency
make benchmark
```

Checklist for publishing a number: record the machine, thread count, `--max-edge`, seed, beam,
model revision and the timestamp from the JSON report. The version strings above are pinned by
commit hash; set `VAS_SEGMENTER_REVISION` / `VAS_CAPTION_REVISION` to freeze them explicitly.

## Files

| Path | Content |
|---|---|
| `artifacts/eval/segmentation_coco-voc.json` | Global metrics, 21×21 confusion, 300 per-image rows |
| `artifacts/eval/segmentation_ade.json` | Global metrics, 150×150 confusion, 300 per-image rows |
| `artifacts/eval/captioning_blip-base.json` | Corpus scores, version, caveats |
| `artifacts/eval/captioning_blip-base_predictions.json` | Raw captions, so scoring is model-free |
| `artifacts/eval/analysis.md` | Auto-generated error-analysis tables |
| `artifacts/benchmark.json` | p50/p95 totals and stage medians |

The `artifacts/` directory is intentionally git-ignored: the scripts reproduce it, and the machine
details in this document are the part that belongs in version control.

## What the data says to try next

Ordered by evidence, not enthusiasm:

1. **Test resolution sensitivity on the two weak VOC classes.** DeepLab hard-caps inference at
   640 px (`MAX_INFERENCE_EDGE` in `segmenter.py`), so raising `--max-edge` alone will not change
   the model input: make the cap configurable first, then re-run the same 300 images at 1024 and
   compare foreground mIoU. `boat` (74 % → background) and `chair` (53 % → background) are the
   classes to watch, and the nine all-background worst images show the same small-object theme.
2. **Merge or post-process ADE siblings for the UI.** `field→grass` (63 %), `rug→floor` (53 %),
   `water→sea` (25 %), `door→wall` (27 %) are taxonomy choices; an app that shows "floor" for a rug
   is not wrong in the way mIoU suggests.
3. **Beam search for captions.** Beam 3–5 with a length penalty is the cheapest experiment against
   the fragment/duplicate failures; keep the same 300 images and report CIDEr/BLEU deltas.
4. **Add Java to the eval image and publish METEOR/SPICE.** One missing runtime is currently the
   only reason two standard metrics are absent.
5. **Scale the sample and add bootstrap intervals** (full 5 000/2 000 images) before comparing
   checkpoints or claiming a fine-tuning win. Keep the 300-image runs for iteration.
