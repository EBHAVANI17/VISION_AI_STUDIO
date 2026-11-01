# Vision AI Studio — End-to-End Implementation & Build Plan

> Target: a production-quality Streamlit product on Hugging Face Spaces, backed by a
> reusable Python core, with a reproducible evaluation harness, LoRA-based fine-tuning,
> and a lightweight MLOps layer (registry, tracking, CI/CD, monitoring).

Current repo state (baseline): one Colab notebook + README. No app, package, tests,
pinned deps, CI, or deployment. Everything below is net-new.

**Status (2026-11-01):** product, CLI, tests, Docker/CI and the evaluation harness are
implemented. [`RESULTS.md`](docs/RESULTS.md) now has 300-image numbers for both segmenters and BLIP
base, with pinned revisions and error analysis. Still open: BLIP large/LoRA and translation/speech
metrics, full-validation runs, fine-tuning (Phase 5), and the MLOps layer (Phase 6).

---

## 1. Decisions

| Area | Decision | Why |
|---|---|---|
| Product | Streamlit app on HF Spaces (Docker SDK) | Fastest path to a live, demoable product; Docker gives full control over deps and model pre-baking |
| Code shape | Thin UI (`app/`) over an importable core package (`src/vision_ai_studio`) | Same core serves the UI, the CLI, the eval scripts, and training scripts; no logic duplicated in notebooks |
| Segmentation | Swappable `Segmenter` interface: **DeepLabV3-R50 COCO/VOC-21** (torchvision) + **SegFormer-B2 ADE20K-150** (HF) | torchvision has no ADE20K weights. VOC-21 is object-centric and matches COCO eval; ADE20K covers generic user uploads (sky, road, wall, tree). Both get evaluated; default in the UI = ADE20K |
| Captioning | BLIP-base (default), BLIP-large (selectable), LoRA adapter (after Phase 5) | Matches prototype; 1 GB / 3 GB RAM respectively; base is the only sane default on 2 vCPU |
| Translation | `Translator` interface: **NLLB-200-distilled-600M** (default, offline, HF) + `deep-translator` (opt-in fallback) | `googletrans==4.0.0-rc1` is unofficial, rate-limited, and breaks on httpx upgrades. NLLB covers all target languages in one cached model |
| TTS | `gTTS` behind a `Synthesizer` interface, audio cached by content hash | Works on HF Spaces (outbound network allowed). Cache prevents repeat calls |
| Training | Pretrained-only fine-tuning (LoRA / head-only). No from-scratch training | From-scratch BLIP needs ~32 A100-days. Out of scope and not the point of the project |
| Model registry | HF Hub repos + pinned revisions | Free, versioned, doubles as artifact storage; no MLflow server to host |
| Experiment tracking | Weights & Biases (free tier) | Lower ops cost than self-hosted MLflow |
| Training compute | Kaggle (30 h/week free GPU) or Colab; never the Space | Space is CPU-only; training lives in notebooks/scripts that call the same `src` modules |

### What runs where

```
GitHub (source of truth)
 ├─ GitHub Actions (CPU)  → lint, mypy, unit/integration tests, docker build, AppTest UI tests
 │                          └─ on tag: sync main → HF Space repo (needs HF_TOKEN secret)
 ├─ HF Space (Docker, CPU 2 vCPU / 16 GB / 50 GB)
 │   └─ Streamlit inference app; models pre-baked into the image at build time
 ├─ Kaggle / Colab (GPU)
 │   └─ notebooks 02/03 call `vision_ai_studio.training.*`; logs → W&B; artifacts → HF Hub
 └─ HF Hub
     ├─ models: Bhavani17/vision-ai-blip-coco-lora, …/vision-ai-segformer-ade20k
     └─ datasets: eval manifests, feedback exports (optional)
```

---

## 2. Target repo layout

```
VISION_AI_STUDIO/
├─ README.md                     # HF Space card frontmatter + full project README
├─ PLAN.md
├─ pyproject.toml                # package + deps + ruff/mypy/pytest config (uv)
├─ uv.lock
├─ requirements-eval.txt         # pycocoevalcap, Java-dependent SPICE extras
├─ Dockerfile                    # python:3.11-slim, CPU torch, pre-baked models
├─ docker-compose.yml
├─ Makefile                      # make dev / test / lint / docker / prefetch
├─ .env.example
├─ .dockerignore  .gitignore  .pre-commit-config.yaml
├─ .github/workflows/ci.yml      # lint + tests + docker build (PR + main)
├─ .github/workflows/deploy.yml  # tag → push to HF Space
├─ app/
│  ├─ streamlit_app.py           # UI only: widgets, layout, session state
│  └─ components.py              # legend, metrics row, download buttons
├─ src/vision_ai_studio/
│  ├─ config.py                  # pydantic-settings: model ids, device, alpha, limits
│  ├─ schemas.py                 # AnalysisResult, Segment, Timings (pydantic v2)
│  ├─ pipeline.py                # analyze(image, config) -> AnalysisResult
│  ├─ cli.py                     # `vas analyze img.jpg -o out/`, `vas prefetch`
│  ├─ models/
│  │  ├─ registry.py             # lazy singletons, revision-pinned loading, device pick
│  │  ├─ segmenter.py            # Segmenter protocol + DeepLabV3VOC + SegFormerADE
│  │  ├─ captioner.py            # Captioner protocol + BLIP (+ optional LoRA adapter)
│  │  └─ translator.py           # Translator protocol + NLLB + GoogleTranslator
│  ├─ viz/palette.py             # deterministic per-class colors, alpha overlay, legend
│  ├─ viz/render.py              # mask→RGBA, colorized mask, side-by-side figure
│  ├─ tts.py                     # Synthesizer protocol + gTTS + in-memory cache
│  ├─ labels/                    # voc21.py, ade150.py (id → name, palette seed)
│  ├─ data/                      # COCO/ADE readers, Karpathy split, transforms
│  └─ training/                  # train_blip_lora.py, train_segformer.py, metrics.py
├─ scripts/
│  ├─ download_coco.py           # val2017 (default) | train2017 | annotations | --stuff
│  ├─ download_ade20k.py
│  ├─ prefetch_models.py         # used by Dockerfile and `make prefetch`
│  ├─ evaluate_segmentation.py   # mIoU / per-class IoU / pixel acc → JSON + MD
│  ├─ evaluate_captioning.py     # BLEU-1..4 / METEOR / ROUGE-L / CIDEr (SPICE optional)
│  ├─ analyze_results.py         # reports → error analysis (classes, confusions, worst images)
│  └─ benchmark.py               # latency p50/p95 by image size, CPU threads
├─ notebooks/                    # outputs stripped, thin wrappers only
│  ├─ 01_exploration.ipynb       # migrated from the current notebook
│  ├─ 02_train_blip_lora.ipynb
│  └─ 03_train_segformer.ipynb
├─ tests/
│  ├─ unit/                      # preprocessing, palette, stats, schemas, translator fake
│  ├─ integration/               # stub models → deterministic pipeline assertions
│  ├─ ui/test_app.py             # streamlit.testing.v1.AppTest
│  ├─ slow/                      # real-model smoke tests (marked, not in PR CI)
│  └─ fixtures/                  # 3–5 small CC-licensed images
├─ configs/
│  ├─ default.yaml  ade20k.yaml  voc21.yaml
│  └─ train/{blip_lora.yaml, segformer_ade.yaml}
├─ data/                         # gitignored, manifests only
├─ artifacts/                    # gitignored (checkpoints, exported figures)
└─ docs/
   ├─ ARCHITECTURE.md  DEPLOYMENT.md  TRAINING.md  EVALUATION.md  RESULTS.md
   └─ screenshots/
```

---

## 3. Core contracts

```python
# schemas.py
class Segment(BaseModel):
    class_id: int
    label: str
    pixels: int
    coverage: float  # 0..1 of image area


class AnalysisResult(BaseModel):
    caption: str
    caption_lang: str  # "en"
    translated: str | None
    target_lang: str | None
    segments: list[Segment]  # sorted by pixels desc, top-N kept
    dominant_label: str
    mask_png_b64: str  # colorized mask, for download
    timings_ms: dict[str, float]
    model_versions: dict[str, str]  # "repo_id@revision"


class Segmenter(Protocol):
    name: str

    def predict(self, rgb: Image.Image) -> tuple[np.ndarray, list[str]]: ...

    # returns HxW uint16 class-id mask at ORIGINAL resolution + id→label list


class Captioner(Protocol):
    def caption(self, rgb: Image.Image, *, beam: int = 1) -> str: ...


class Translator(Protocol):
    def translate(self, text: str, target: str) -> str: ...


class Synthesizer(Protocol):
    def speak(self, text: str, lang: str) -> bytes: ...  # mp3
```

Rules baked into the core:
- Every model load goes through `registry.py`; nothing loads at import time.
- `torch.inference_mode()` and `torch.set_num_threads(min(4, os.cpu_count()))`.
- All preprocessing handles EXIF transpose, RGB coercion, `Image.MAX_IMAGE_PIXELS` guard,
  and a max input edge (default 2048 px) with the mask upsampled back to display size.
- Segmentation normalization is per-model (DeepLab: ImageNet mean/std, 513×513 resize;
  SegFormer: ImageNet mean/std, 512×512 pad-to-square; BLIP: its own processor).
- Mask upsampling is bilinear on logits, then `argmax` — never nearest on class ids.

---

## 4. Phases, deliverables, exit criteria

### Phase 0 — Security & repo hygiene (≈2 h)

1. **Rotate the leaked ngrok authtoken** in the ngrok dashboard (it is public in cell 9 of the
   notebook). Treat the old token as compromised.
2. Purge it from git history (`git filter-repo --replace-text`) and force-push *after*
   explicit approval; the repo has one commit, so a history rewrite is cheap.
3. Add `detect-secrets` + `nbstripout` to pre-commit so this cannot recur.
4. Strip notebook outputs (4.9 MB → ~30 KB), move to `notebooks/01_exploration.ipynb`.
5. Add `.gitignore`, `.env.example`, `LICENSE`, `Makefile`, `pyproject.toml`, ruff/mypy config.
6. Rewrite README: badges, architecture diagram, live demo link (placeholder), honest
   scope statement, model/data licenses.

**Exit:** `pre-commit run --all-files` clean; no secret anywhere in `git log -p`; notebook < 100 KB.

---

### Phase 1 — Core engine (≈3–4 d)

1. `schemas.py`, `config.py` (`SEG_MODEL=ade|voc`, `CAPTION_MODEL=blip-base|blip-large|<hf-repo>`,
   `TRANSLATOR=nllb|google|none`, `DEVICE=auto`).
2. `models/segmenter.py`:
   - `DeepLabV3VOC`: `deeplabv3_resnet50(weights=DEFAULT)`, 21 VOC labels.
   - `SegFormerADE`: `nvidia/segformer-b2-finetuned-ade-512-512`, 150 ADE labels.
   - Shared postprocess: logits → upsample → argmax → `Segment` list → coverage.
3. `models/captioner.py`: BLIP base/large, greedy default, `beam=3` option.
   Supports loading a PEFT adapter (`LORA_REPO` env) — implemented in Phase 5, wired now.
4. `models/translator.py`: NLLB-200-distilled-600M (lazy; only loaded if a non-`en` target is
   chosen), language code map (en/hi/ta/te/kn/ml/fr/es/zh), `google` fallback via
   `deep-translator`, `none` for tests.
5. `viz/`: deterministic palette (hash label → stable color, colorblind-safe for top classes),
   0.45-alpha overlay, mask PNG, legend HTML/PNG.
6. `tts.py`: gTTS + `lru_cache` keyed by `sha256(text|lang)`.
7. `pipeline.py`: orchestration + per-stage timings; partial-failure semantics (a translator
   outage must not lose the caption).
8. `cli.py`: `vas analyze`, `vas prefetch`, `vas info` — proves the core is UI-independent.

**Exit:** `vas analyze tests/fixtures/*.jpg` works on CPU; unit + stub-integration tests green;
benchmark: 1024 px image ≤ 6 s p95 on 2 vCPU (base models, no translation), ≤ 12 s with NLLB.

---

### Phase 2 — Streamlit product (≈2–3 d)

- `@st.cache_resource` for models; `@st.cache_data(max_entries=64)` for
  `analyze(hash(image_bytes), cfg, alpha)`. No model reload on rerun.
- Layout: header card → uploader/paste → sample gallery → sidebar (model, palette alpha,
  top-k classes, target language, beam) → tabs **Overview / Segmentation / Caption & Audio**.
- Outputs: overlay, colorized mask, top-8 class table with % coverage and bars, caption,
  translation, audio player, and a **JSON download** of `AnalysisResult` + PNG downloads.
- Correctness/perf details: `use_container_width=True` (not the deprecated
  `use_column_width`), `st.dataframe(hide_index=True)`, `ImageOps.exif_transpose`, cap upload at
  15 MB, 10 MP guard, friendly errors (corrupt file, unsupported format, model OOM).
- Streamlit `AppTest` test: upload fixture → click Analyze → assert caption tab non-empty and
  download buttons present (models stubbed via monkeypatch).
- Replace the notebook's "green channel = class id" overlay hack with real per-class colors.

**Exit:** app runs locally via `make dev`; AppTest green; no rerun-triggered model reloads;
UI handles the three failure modes without tracebacks.

---

### Phase 3 — Packaging & deployment (≈1–2 d)

- `pyproject.toml` + `uv.lock`; `requirements-eval.txt` isolated (pycocoevalcap needs Java for SPICE).
- `Dockerfile` (multi-stage):
  - base `python:3.11-slim`, CPU wheels from `download.pytorch.org/whl/cpu`;
  - `RUN python -m vision_ai_studio.cli prefetch` → models baked in, `HF_HOME=/app/.cache/huggingface`;
  - non-root user, `HEALTHCHECK CMD curl -f http://localhost:7860/_stcore/health`;
  - target image < 6 GB, cold start < 20 s.
- `docker-compose.yml` for local parity; `make docker`.
- HF Docker Space: README frontmatter (`sdk: docker`, `app_port: 7860`, title/emoji/colors),
  `streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0`,
  `--server.headless true`, telemetry off.
- CI (`ci.yml`): uv sync → ruff → mypy → pytest (unit+integration+AppTest) → `docker build`.
  Actions pinned to commit SHAs; `permissions: contents: read`.
- Deploy (`deploy.yml`): on `v*` tag, push to the Space repo using `HF_TOKEN` secret (write token
  stored only in GitHub secrets); Space rebuilds from Dockerfile.
- `docs/DEPLOYMENT.md`: one-time setup (create Space, add remote, secrets), rollback by re-tagging.

**Exit:** public Space URL loads, analyzes a sample in < 15 s, survives restart with no
model re-download; CI green on a clean clone.

---

### Phase 4 — Evaluation harness (≈2–3 d)

- `download_coco.py`: `val2017` images + `annotations_trainval2017` (~1.2 GB) by default;
  `train2017` (18 GB) and COCO-Stuff annotations behind flags. Verify SHA-256.
- **Segmentation (VOC-21 / DeepLabV3)** on COCO val2017: build semantic GT from
  `instances_val2017.json` using the torchvision VOC↔COCO category mapping; images whose GT
  contains only unmapped categories are skipped and counted in the report. Report mIoU,
  per-class IoU, pixel accuracy, and coverage (#images evaluated).
- **Segmentation (ADE20K / SegFormer)** on the ADE20K val split (2 000 images): mIoU + per-class.
- **Captioning** on COCO val2017: 5 000 images (or a fixed 1 000-image manifest for speed),
  `pycocoevalcap` → BLEU-1..4, METEOR, ROUGE-L, CIDEr; SPICE behind `--spice` + Java.
  Compare BLIP-base / BLIP-large / LoRA adapter in one table.
- Determinism: fixed image manifests (`manifests/coco_val_1000.json` with IDs + hashes), fixed seeds,
  raw per-image outputs cached under `artifacts/eval/` so re-scoring is offline.
- `analyze_results.py` → error analysis (weakest classes, confusions, ratios, worst images);
  `docs/RESULTS.md` holds the curated tables, latency table and caveats.
- Runtime test: eval scripts run on 50 images in CI-skipped mode (`--limit 50`), verified once
  manually; slow marker for full runs.

**Exit:** `docs/RESULTS.md` has committed numbers for all six model configurations, with
dataset split, sample counts, and caveats stated (unmapped COCO classes, CPU-only timings).

---

### Phase 5 — Fine-tuning (≈3–5 d dev + 10–30 GPU h)

Data:
- Captions: COCO **Karpathy split** (113 k train / 5 k val / 5 k test), built by
  `data/karpathy.py`; optional Visual Genome / Flickr30k later.
- Segmentation: COCO-Stuff 164 k (reuses COCO train2017 images, adds 171-class stuff+thing labels);
  optional ADE20K 25 k.

1. **BLIP LoRA** (`training/train_blip_lora.py`, config `configs/train/blip_lora.yaml`):
   `peft` LoRA on the text decoder attention/FFN projections (vision tower frozen; documented
   alternative to also adapt it), r=16, α=32, dropout 0.05, bf16/fp16, AdamW lr 1e-4,
   cosine + 5 % warmup, effective batch 64 (micro-batch 8–16 × grad-accum), 1–2 epochs,
   image 384. Start on a 30 k subset to validate the loop, then full 113 k.
   Eval every 1 k steps on the Karpathy val split (CIDEr primary); early stop on plateau.
2. **SegFormer head/decoder fine-tune** (`train_segformer.py`): `SegformerForSemanticSegmentation`,
   freeze encoder for epoch 1, then unfreeze with lr/10; 512² crops, batch 16 on A100 / 4×accum on T4;
   mIoU on ADE/COCO-Stuff val. Baseline gate: must beat the pretrained checkpoint's mIoU.
3. Notebooks 02/03 import these modules and only add Colab/Kaggle plumbing. No duplicated logic.
4. Logging: W&B project `vision-ai-studio`; log config hash, git SHA, data manifest hash, metrics,
   and sample predictions.
5. Artifacts: push adapter/checkpoint + model card to HF Hub; record the **revision SHA** in
   `configs/` and in the app's `model_versions` output.
6. Promotion gate: fine-tuned model becomes the app default only if it beats baseline on the
   held-out manifest and stays within the latency budget; otherwise it remains a UI option.

**Exit:** at least one fine-tuned captioner beats BLIP-base on CIDEr on the 1 000-image manifest,
or the negative result is documented with numbers; both checkpoints reproducible from config + SHA.

---

### Phase 6 — MLOps (≈2–3 d)

- **Registry:** HF Hub repos + revision pinning; model cards with metrics, data, license;
  `scripts/promote.py <repo>@<rev>` updates `configs/default.yaml` and opens a PR.
- **Tracking:** W&B on every training run (Phase 5); run summary links stored in the model card.
- **CI/CD:** PR → lint/mypy/tests/docker build; merge → Space sync on tag; manual `workflow_dispatch`
  training job that only *launches* (GPU work runs on Kaggle/Colab, artifacts land in HF Hub).
- **Monitoring:** structured JSON logs to stdout (model ids, latency, error class, image size) —
  HF captures them; optional Sentry for exceptions; no image content logged.
- **Feedback:** opt-in 👍/👎 per analysis; stored in a small HF Dataset (append-only) via a
  write-scoped token in Space secrets; a `review_feedback.py` script produces a triage sheet.
- **Data/version hygiene:** manifests with image IDs + hashes; `scripts/verify_manifest.py`;
  DVC deliberately omitted (data lives in COCO/ADE and HF Hub; manifests + hashes suffice).
- **Rollback runbook** in `docs/DEPLOYMENT.md`: revert config revision → re-tag → Space rebuild.
- **Dependency/security:** Dependabot, `pip-audit` in CI, pinned lockfile, container runs as non-root.

**Exit:** one model promotion done end-to-end through the registry script; a failing CI check
blocks a bad config; Space logs show model versions + latency for real traffic.

---

### Phase 7 — Hardening & polish (≈1–2 d)

- Performance: benchmark ONNX Runtime / INT8 dynamic quantization for DeepLabV3 and SegFormer;
  adopt only if it holds mIoU within 0.5 and cuts p95 ≥ 30 %. Do not export BLIP unless measured.
- Security: upload size/type checks, PIL decompression-bomb guard, no arbitrary path input in the
  CLI beyond local files, no secrets in the image, least-privilege Space token.
- UX: keyboard flow, sample images, dark-mode-safe palette, alt text, loading skeletons,
  clear "running on CPU — expect a few seconds" copy, demo GIF in README.
- Docs: `ARCHITECTURE.md` (with ASCII diagram), `TRAINING.md`, `EVALUATION.md`,
  `DEPLOYMENT.md`, troubleshooting, license/attribution section.
- Tag `v1.0.0`; write a short release note with the results table.

**Exit:** public URL, green CI, results table, GIF, and a clean-clone `make dev` within 10 minutes.

---

## 5. Testing strategy

| Layer | Scope | Where |
|---|---|---|
| Unit | preprocessing shapes/EXIF, palette determinism, mask→stats, schema serialization, language map, TTS cache key | PR CI |
| Integration | pipeline with stub segmenter/captioner/translator: exact mask, exact stats, translator failure degrades gracefully | PR CI |
| UI | `AppTest` upload → analyze → tabs/downloads; error paths | PR CI |
| Slow smoke | real DeepLab + BLIP + SegFormer on 3 fixtures; caption non-empty, coverage sums to 1.0, dominant class in expected set | nightly / manual |
| Perf | `benchmark.py` p50/p95 per size × model; threshold recorded, not gated | manual, results committed |
| Eval | fixed manifests + cached outputs; score-only path must not need model downloads | manual |

No test downloads model weights in PR CI; HF cache is pre-populated only in the Docker build.

---

## 6. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Leaked ngrok token in public history | Account abuse | Rotate now; history purge; `detect-secrets` in pre-commit |
| `googletrans` unreliable | Translation/TTS breaks in prod | NLLB as default; interface + fake for tests |
| Free Space (2 vCPU) latency | Demo feels slow | Base models by default; pre-baked weights; caching; translation lazy; show timings |
| Space cold start / restart | First request slow | Models baked into image; `_stcore/health` check; keep image < 6 GB |
| COCO GT for segmentation is instance-only | Misleading mIoU | Map to VOC-21, skip/report unmapped images explicitly |
| SPICE needs Java | Eval setup friction | Isolated `requirements-eval.txt`; SPICE optional |
| Fine-tuning gains may be small | Overpromised results | Promotion gate + honest reporting of negative results |
| BLIP-large on CPU | 10–20 s/request, 3 GB RAM | UI warning; not the default; optional |
| Model licenses (SegFormer/Mask2Former/BLIP differ) | Legal exposure if commercialized | Document each license in README + model cards; check before any commercial use |
| HF upload size limits / private weights | Deploy failures | Small weights only; adapters are tiny; keep large checkpoints out of the image |

---

## 7. Effort

| Phase | Time | Gate |
|---|---|---|
| 0 Hygiene | 2 h | No secrets, clean history |
| 1 Core engine | 3–4 d | CLI works, tests green, latency budget met |
| 2 Streamlit UI | 2–3 d | AppTest green, no reloads |
| 3 Docker + HF Space | 1–2 d | **Public URL** |
| 4 Evaluation | 2–3 d | RESULTS.md committed |
| 5 Fine-tuning | 3–5 d + 10–30 GPU h | Promotion gate |
| 6 MLOps | 2–3 d | Registry + monitor in place |
| 7 Hardening | 1–2 d | v1.0.0 |

≈3 weeks part-time. Ship Phase 3 first so the live demo exists early, then layer 4–6.

---

## 8. First three PRs

1. **PR-1 (hygiene + skeleton):** token rotation/purge, pre-commit, package skeleton, schemas,
   config, CLI stub, fixtures, unit tests, CI. No model downloads.
2. **PR-2 (MVP product):** segmenters + captioner + palette + pipeline + Streamlit UI + AppTest +
   Dockerfile + HF Space deploy → **live URL**.
3. **PR-3 (translation/TTS + eval harness):** NLLB + gTTS, COCO/ADE download scripts, eval scripts,
   `docs/RESULTS.md` with first numbers.

Then Phase 5 (training) and Phase 6 (MLOps) as separate tracks.

---

## 9. Definition of done

- Public HF Space: upload → caption + translation + audio + segmentation overlay + class table +
  JSON/PNG downloads, < 15 s p95 with translation on 2 vCPU.
- Core package tested, typed, importable without Streamlit; CLI parity with the UI.
- Numbers in `docs/RESULTS.md` for 3 captioners × 2 languages-worth of metrics and 2 segmenters,
  with sample counts and caveats.
- At least one fine-tuned model (or a documented negative result) promoted through the registry
  script, pinned by revision.
- Green CI, pre-commit, Dependabot; no secrets in history; rollback documented.
