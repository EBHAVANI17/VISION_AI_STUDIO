# Vision AI Studio

**Image segmentation · captioning · translation · speech, in one pass.**

Upload a photo and get a segmentation overlay with a ranked class breakdown, a natural-language
caption, a translation into nine languages, and spoken audio. The app runs on CPU, needs no API
keys, and keeps images in memory only.

[![CI](https://github.com/EBHAVANI17/VISION_AI_STUDIO/actions/workflows/ci.yml/badge.svg)](https://github.com/EBHAVANI17/VISION_AI_STUDIO/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)

Live demo: _add the Hugging Face Space URL here after the first deploy._

---

## What it does

| Stage | Model | Output |
|---|---|---|
| Segmentation | **SegFormer-B2** (ADE20K, 150 classes) or **DeepLabV3-R50** (COCO/VOC, 21 classes) | Color overlay, colorized mask, ranked class table with pixel counts and coverage |
| Captioning | **BLIP** base or large (optional LoRA adapter) | One-sentence description |
| Translation | **NLLB-200 distilled 600M** (offline) or Google Translate (opt-in) | Hindi, Tamil, Telugu, Kannada, Malayalam, French, Spanish, Chinese |
| Speech | **gTTS** | MP3 of the translated or original caption |

Everything is downloadable as PNG (overlay, mask) and JSON (full result, model versions, timings).

## Measured results

Frozen checkpoints, CPU, 300-image random samples (seed 0), pinned revisions — full protocol,
per-class tables and failure analysis in [`docs/RESULTS.md`](docs/RESULTS.md):

| Task | Model | Dataset | Result |
|---|---|---|---|
| Segmentation | DeepLabV3-R50 (COCO/VOC-21) | COCO val2017 | foreground mIoU **0.5947**, pixel acc 0.9269 |
| Segmentation | SegFormer-B2 (ADE20K-150) | ADE20K val | mIoU **0.4480**, pixel acc 0.8150 |
| Captioning | BLIP base | COCO val2017 | CIDEr **0.8266**, BLEU-4 0.2555, ROUGE-L 0.4375 |
| Latency | BLIP + SegFormer-B2 | 512→2048 px | p50 **6.5–9.7 s** on 4 CPU threads |

The evaluation also produces an error-analysis report (weakest classes, truth→prediction
confusions, over/under-segmentation, worst images, caption failure modes), so the numbers come with
an explanation of where the models break.

## Architecture

The Streamlit app is a thin layer over a reusable core. The same core powers the CLI, the tests
and the evaluation scripts, so there is no logic locked inside notebooks.

```
app/streamlit_app.py        UI only: widgets, caching, layout
        |
        v
src/vision_ai_studio/
  pipeline.py               caption -> segment -> translate -> render
  models/registry.py        lazy, process-wide model singletons
  models/segmenter.py       SegFormer-ADE150, DeepLabV3-VOC21
  models/captioner.py       BLIP base/large (+ optional PEFT adapter)
  models/translator.py      NLLB, Google, null
  viz/                      deterministic palette, overlay + mask rendering
  stats.py                  mask -> ranked segments
  analysis.py               evaluation helpers: confusions, worst images, token F1
  tts.py                    gTTS with content-hash caching
  cli.py                    `vas analyze | prefetch | info`
        |
        v
scripts/                    dataset download, evaluation, benchmarking
```

Design notes:

- **Lazy weights.** Nothing loads at import. Fake models (`VAS_MODELS=fake`) make the UI and tests
  run without downloading a byte.
- **Lazy translation.** NLLB is only fetched when the first non-English request arrives, so the
  default English path stays light.
- **Degrade, don't crash.** A caption or translation failure produces a warning and keeps the rest
  of the result. A segmentation failure is fatal, because there is nothing meaningful left to show.
- **Stable colors.** Class colors are derived from a hash of `(class_id, label)`, so the same class
  is the same color across runs and machines.

## Quickstart

```bash
git clone https://github.com/EBHAVANI17/VISION_AI_STUDIO.git
cd VISION_AI_STUDIO

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[ui,dev]"

streamlit run app/streamlit_app.py
```

First analysis downloads ~1.3 GB of weights (BLIP base plus the selected segmenter). Weights are
cached by Hugging Face and reused on later runs. Translation weights (~2.4 GB) are downloaded the
first time you choose a non-English language.

No GPU is required. On a 4-thread CPU, expect roughly 5–10 s for BLIP captioning, 5–20 s for
segmentation and 1–2 s for rendering. The first non-English request also loads NLLB (~2.4 GB),
which dominates that request (a minute or more on CPU); subsequent translations reuse the loaded
model.

### Try the UI without downloading models

```bash
VAS_MODELS=fake streamlit run app/streamlit_app.py
```

`fake` mode uses deterministic stand-ins for every model, which is what the UI tests use.

## Configuration

All settings are environment variables read at startup (see [`.env.example`](.env.example)):

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `VAS_SEGMENTER` | `ade`, `voc` | `ade` | SegFormer-ADE150 or DeepLabV3-VOC21 |
| `VAS_CAPTIONER` | `blip-base`, `blip-large` | `blip-base` | Captioning backbone |
| `VAS_TRANSLATOR` | `nllb`, `google`, `none` | `nllb` | Translation backend |
| `VAS_DEVICE` | `auto`, `cpu`, `cuda`, `mps` | `auto` | Inference device |
| `VAS_MODELS` | `real`, `fake` | `real` | Real weights or deterministic fakes |
| `VAS_LORA_REPO` | HF repo id | unset | LoRA captioning adapter |
| `VAS_CAPTION_REVISION` | HF revision | unset | Pin the captioner revision |
| `VAS_SEGMENTER_REVISION` | HF revision | unset | Pin the segmenter revision |
| `VAS_MAX_EDGE` | 256–4096 | `2048` | Longest input edge before inference |
| `VAS_MAX_UPLOAD_MB` | 1–100 | `15` | Upload size cap |
| `VAS_OVERLAY_ALPHA` | 0–1 | `0.45` | Overlay opacity |

## CLI

```bash
vas info
vas analyze photo.jpg -o outputs --lang hi
vas analyze photo.jpg --seg voc --caption blip-large --translator none
vas prefetch --seg ade --caption blip-base
```

`analyze` writes `<name>.json`, `<name>.overlay.png` and `<name>.mask.png` and prints a summary:

```
Image:      1280x853
Caption:    two cats sleeping on a pink couch
Dominant:   cat
Classes:    cat 41.2% | couch 22.4% | background 18.9%
Timings:    caption 1340 ms | segment 890 ms | translate 2100 ms
```

The CLI and the UI share the same pipeline, so anything reproducible in one is reproducible in the
other.

## Testing

```bash
pytest -m "not slow"      # unit, integration, CLI and Streamlit AppTest — no weights
VAS_RUN_SLOW=1 pytest tests/slow -m slow   # real BLIP + both segmenters
```

The fast suite runs the whole pipeline with fake models, including failure injection (caption
offline, translator offline, segmenter crash) and the Streamlit upload → analyze flow.

`pre-commit install` wires up ruff, secret scanning (`.secrets.baseline`), notebook output stripping
and large-file checks.

## Docker

```bash
docker build -t vision-ai-studio .
docker run --rm -p 7860:7860 vision-ai-studio
```

The image bakes BLIP and both segmenters at build time and runs as a non-root user with a
`/_stcore/health` healthcheck. To include translation weights (adds ~2.4 GB):

```bash
docker build --build-arg PREFETCH_TRANSLATOR=nllb -t vision-ai-studio .
```

## Deploying to Hugging Face Spaces

The Space is a **Docker** SDK Space. The image listens on port `7860`; the Space card lives in
[`space/README.md`](space/README.md).

1. Create a Docker Space, e.g. `EBHAVANI17/vision-ai-studio`.
2. In GitHub, add the secret `HF_TOKEN` (write scope) and the repository variable `HF_SPACE`
   (`EBHAVANI17/vision-ai-studio`).
3. Push a tag: `git tag v0.1.0 && git push origin v0.1.0`. The deploy workflow copies the Space
   card, replaces the README and force-pushes the tree to the Space, which rebuilds the image.

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for manual deployment, rollback and troubleshooting.

## Evaluation

The harness evaluates both segmentation backends and the captioner on the datasets each model was
trained for, then turns the reports into an error analysis. See
[`docs/EVALUATION.md`](docs/EVALUATION.md) for the protocol and [`docs/RESULTS.md`](docs/RESULTS.md)
for the measured numbers and caveats.

```bash
python scripts/download_coco.py --dest data/coco --splits val2017 --annotations

make eval-seg       # both segmenters, 300 images, seed 0, max-edge 640
make eval-caption   # BLIP base, 300 images, beam 1
make eval-report    # artifacts/eval/analysis.md
make benchmark      # CPU latency at 512/1024/2048 px
```

Reports include the full confusion matrix, per-image rows and the pinned model revision, so the
analysis can be re-run without inference.

## Models and licenses

| Component | Checkpoint | License |
|---|---|---|
| Captioning | `Salesforce/blip-image-captioning-base` | BSD-3-Clause |
| Segmentation (VOC) | torchvision `deeplabv3_resnet50` COCO weights | BSD-3-Clause (torchvision) |
| Segmentation (ADE) | `nvidia/segformer-b2-finetuned-ade-512-512` | check the model card before commercial use |
| Translation | `facebook/nllb-200-distilled-600M` | CC-BY-NC-4.0 (non-commercial) |
| Speech | gTTS (Google Translate TTS endpoint) | see gTTS/Google terms |

The code in this repository is MIT. Model weights keep their own licenses; review them before any
commercial deployment. COCO images and annotations are not redistributed here.

## Project status

- [x] Core pipeline with two segmenters, BLIP, NLLB, gTTS
- [x] Streamlit app with caching, downloads, error handling and audio
- [x] CLI with `analyze`, `prefetch`, `info`
- [x] Docker image, CI, HF Space deploy workflow
- [x] Fast test suite with fakes and UI tests
- [x] COCO/ADE evaluation and benchmark scripts
- [x] 300-image evaluation of both segmenters and BLIP with pinned revisions and error analysis
- [ ] Full-validation evaluation and before/after fine-tuning metrics (next)
- [ ] LoRA fine-tuning notebooks + scripts (next)
- [ ] Model registry promotion flow, experiment tracking, monitored Space (next)

See [`PLAN.md`](PLAN.md) for the full roadmap and exit criteria.

## Security notes

- No credentials live in this repository. The original notebook's ngrok token was removed from the
  working tree; rotate any token that has ever been committed.
- Uploads are validated, pixel-capped (40 MP) and processed in memory; nothing is written to disk.
- Container runs as uid 1000; pull requests run lint, type checks, tests and a container
  healthcheck in CI, and Dependabot keeps dependencies and actions current.
