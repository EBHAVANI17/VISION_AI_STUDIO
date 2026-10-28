# Architecture

## Goals

1. One inference core used by the UI, the CLI and the evaluation scripts.
2. No heavyweight imports or downloads until they are actually needed.
3. Failures degrade locally (a broken translator must not hide the caption).
4. Reproducible outputs: stable colors, pinned model revisions, JSON results.

## Request flow

```
upload bytes
   |
   v
image.load_image        size cap, EXIF orientation, RGB, downscale to max_edge
   |
   v
pipeline.analyze
   |-- captioner.caption(image)            BLIP (base/large, optional LoRA)
   |-- segmenter.predict(image)            SegFormer-ADE150 or DeepLabV3-VOC21
   |       -> mask (HxW int32), labels, version
   |-- stats.segment_stats(mask)           ranked Segment list, coverage
   |-- viz.render.render_pngs(image)       overlay PNG + colorized mask PNG
   |-- translator.translate(caption)       NLLB (lazy) / Google / none
   v
AnalysisResult (frozen dataclass)
   |
   +--> Streamlit tabs, downloads, audio
   +--> CLI JSON + PNGs
   +--> eval scripts (mask only)
```

Audio is deliberately outside the pipeline: `tts.synthesize_mp3(text, lang)` is called on demand,
cached by `(text, lang)`, so a speech failure never invalidates an analysis.

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` (frozen dataclass), env parsing, validation, cache key |
| `image.py` | decode, validate, orient, downscale, hash, PNG encode |
| `pipeline.py` | orchestration, timings, partial-failure policy, model assembly |
| `schemas.py` | `Segment`, `AnalysisResult`, JSON payload |
| `stats.py` | mask → segments, dominant label |
| `viz/palette.py` | deterministic class colors, background neutral |
| `viz/render.py` | mask alignment, colorized mask, alpha overlay |
| `models/base.py` | `Segmenter`/`Captioner`/`Translator` protocols, error types |
| `models/segmenter.py` | lazy-import torchvision DeepLabV3 and HF SegFormer |
| `models/captioner.py` | lazy-import BLIP, optional PEFT adapter |
| `models/translator.py` | NLLB, Google, null; language normalisation |
| `models/registry.py` | process-wide `lru_cache` singletons, device resolution, lazy translator |
| `models/fakes.py` | deterministic stand-ins for tests and UI development |
| `tts.py` | gTTS wrapper with cache |
| `cli.py` | `info`, `prefetch`, `analyze` |

## Contracts

```python
class Segmenter(Protocol):
    name: str

    def predict(self, image: Image.Image) -> Segmentation: ...  # mask at original size


class Captioner(Protocol):
    name: str

    def caption(self, image: Image.Image, *, beam: int = 1) -> str: ...


class Translator(Protocol):
    name: str
    version: str

    def translate(self, text: str, target: str) -> str: ...
```

`AnalysisResult` is frozen and picklable; Streamlit caches it directly. Binary artifacts (PNG
bytes) live in the result, but `to_payload()` / `to_json()` exclude them so API responses stay
small.

## Error policy

| Failure | Behaviour |
|---|---|
| Invalid/corrupt/oversized upload | `ImageError`, surfaced to the user, no analysis |
| Captioner failure | empty caption + warning; segmentation continues |
| Translator failure | `translated=None` + warning; caption preserved |
| Segmentation failure | `ModelInferenceError` propagates; no result is fabricated |
| TTS failure | error shown in the speech panel; analysis untouched |

## Caching

| Layer | Mechanism | Key |
|---|---|---|
| Model weights | `lru_cache` in `models/registry.py` | `(name, device, revision, mode)` |
| Streamlit model objects | `st.cache_resource` | `Settings` |
| Analysis results | `st.cache_data(max_entries=8)` | image bytes + `Settings` + language/beam/alpha |
| Speech | `lru_cache(128)` | `(text, lang)` |

The UI passes the whole frozen `Settings` object; its `cache_key()` ignores layout-only fields
(opacity, upload cap) where relevant.

## Performance notes

- Inputs are downscaled to `VAS_MAX_EDGE` (default 2048) before inference; segmenters internally
  work at ≤640 px (DeepLab) or 512² (SegFormer), then upsample logits to the display size.
- Logits are upsampled with bilinear interpolation *before* `argmax`; masks are never resized with
  nearest-neighbour class ids except for already-computed masks.
- `torch.inference_mode()` and 4 CPU threads are used by default.
- The translation model is lazy: an English-only session never downloads NLLB.

## Extension points

- **New segmenter:** implement `predict`, return a `Segmentation`, register it in
  `models/segmenter.build_segmenter` and add the key to `config.SEGMENTERS`.
- **New captioner:** same shape; return a `version` string for the result metadata.
- **New translator:** implement `translate` and add a branch in `build_translator`.
- **New renderer:** replace `viz.render.render_pngs`; the pipeline only needs two PNGs.

## Fake mode

`VAS_MODELS=fake` swaps in `FakeSegmenter`, `FakeCaptioner` and `FakeTranslator`. It exists so the
UI, AppTest and CLI tests run in under a second with no network or weights. It is not a fallback
for production: the UI shows a banner and disables speech in this mode.
