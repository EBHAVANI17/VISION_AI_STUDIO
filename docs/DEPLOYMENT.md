# Deployment

## Local

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[ui,dev]"
streamlit run app/streamlit_app.py
```

The app is at http://localhost:8501. Models are cached under the default Hugging Face cache
(`~/.cache/huggingface`), so the download happens once.

To work on the UI without weights:

```bash
VAS_MODELS=fake streamlit run app/streamlit_app.py
```

## Docker

```bash
docker build -t vision-ai-studio .
docker run --rm -p 7860:7860 vision-ai-studio
# http://localhost:7860
```

What the image does:

- installs CPU-only PyTorch from `download.pytorch.org/whl/cpu`;
- bakes BLIP base, DeepLabV3-VOC21 and SegFormer-ADE150 into `HF_HOME`;
- runs as uid 1000;
- exposes `/_stcore/health` as a Docker healthcheck.

Translation weights are *not* baked by default (~2.4 GB). To include them:

```bash
docker build --build-arg PREFETCH_TRANSLATOR=nllb -t vision-ai-studio .
```

`docker compose up --build` runs the same image on port 7860.

## Hugging Face Spaces (Docker SDK)

The Space repository is the deployment target; GitHub remains the source of truth.

### One-time setup

1. Create a **Docker** Space, e.g. `EBHAVANI17/vision-ai-studio`.
2. Create a write-scoped Hugging Face token (<https://huggingface.co/settings/tokens>).
3. In the GitHub repository:
   - add secret `HF_TOKEN` (the write token);
   - add variable `HF_SPACE` = `EBHAVANI17/vision-ai-studio`.
4. Push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `Deploy to Hugging Face Spaces` workflow copies `space/README.md` to the root README (the
Space card frontmatter), initialises a fresh repository and force-pushes the tree to the Space.
The Space then builds the `Dockerfile` and starts the app on port 7860.

### Manual deploy

```bash
git clone https://huggingface.co/spaces/EBHAVANI17/vision-ai-studio space-checkout
rsync -a --exclude .git --exclude .venv --exclude data --exclude artifacts \
      --exclude tests --exclude notebooks --exclude .github ./ space-checkout/
cp space/README.md space-checkout/README.md
cd space-checkout && git add -A && git commit -m "deploy" && git push
```

Force-pushing from CI rewrites the Space history. That is normal for generated deployments, but
anything pushed directly to the Space and not present in GitHub will be lost.

### Rollback

The Space runs whatever commit is at its `main`. To roll back, check out the previous Git tag in
this repository and re-run the deploy workflow (`workflow_dispatch`), or push the older tree to the
Space manually. Model revisions can be pinned independently with `VAS_CAPTION_REVISION` and
`VAS_SEGMENTER_REVISION`.

## Configuration in production

Set variables in the Space settings (Settings → Variables and secrets). Avoid secrets in the
Dockerfile; nothing in the image requires a token.

Recommended Space settings:

| Variable | Value |
|---|---|
| `VAS_SEGMENTER` | `ade` |
| `VAS_CAPTIONER` | `blip-base` |
| `VAS_TRANSLATOR` | `nllb` |
| `VAS_DEVICE` | `cpu` |
| `VAS_MAX_EDGE` | `2048` |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails at `pip install torch` | arm64 runner without a CPU wheel | build on x86_64 or install the arm64 wheel index |
| Build OOM | model prefetch at build | remove one `prefetch` call or build with more memory |
| App starts but first analysis is slow | lazy model move to RAM | expected; subsequent requests use the cached models |
| Translation hangs on first non-English request | NLLB download (~2.4 GB) | rebuild with `PREFETCH_TRANSLATOR=nllb` or warn users in the UI |
| Container healthy, page 502 | reverse proxy/port mismatch | ensure `app_port: 7860` and `--server.address=0.0.0.0` |
| `OSError: No space left` | 50 GB Space disk exhausted | keep large checkpoints out of the image; prune caches |
| Blank page after deploy | README frontmatter missing on the Space | copy `space/README.md` to the Space root |

## Resource profile

| Resource | Value |
|---|---|
| Image size | ~3.5 GB (models included) |
| RAM at idle | ~1.2 GB |
| RAM with BLIP + SegFormer + NLLB | ~5–6 GB |
| Free Space tier | 2 vCPU, 16 GB RAM, 50 GB disk |
