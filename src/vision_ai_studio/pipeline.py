"""The end-to-end pipeline: caption -> segment -> translate -> render."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from PIL import Image

from vision_ai_studio.config import Settings
from vision_ai_studio.models.base import Captioner, ObjectDetector, Segmenter, Translator
from vision_ai_studio.models.registry import (
    LazyTranslator,
    get_captioner,
    get_segmenter,
    resolve_device,
)
from vision_ai_studio.models.translator import normalize_lang
from vision_ai_studio.schemas import AnalysisResult
from vision_ai_studio.stats import dominant_label, segment_stats
from vision_ai_studio.viz.render import align_mask, render_pngs


@dataclass(frozen=True)
class Models:
    segmenter: Segmenter
    captioner: Captioner
    translator: Translator | None = None
    detector: ObjectDetector | None = None


def build_models(settings: Settings) -> Models:
    """Load (or fetch from the process cache) the models described by ``settings``."""
    device = "cpu" if settings.models == "fake" else resolve_device(settings.device)
    segmenter = get_segmenter(
        settings.segmenter, device, settings.segmenter_revision, settings.models
    )
    captioner = get_captioner(
        settings.captioner,
        device,
        settings.caption_revision,
        settings.lora_repo,
        settings.models,
    )
    translator: Translator | None = None
    if settings.translator != "none":
        translator = LazyTranslator(
            settings.translator,
            device="cpu" if settings.translator == "google" else device,
            mode=settings.models,
        )
    return Models(segmenter=segmenter, captioner=captioner, translator=translator)


def analyze(
    image: Image.Image,
    *,
    models: Models,
    target_lang: str = "en",
    overlay_alpha: float = 0.45,
    beam: int = 1,
) -> AnalysisResult:
    """Run the full pipeline. Caption/translation failures degrade; segmentation failures raise."""
    lang = normalize_lang(target_lang)
    warnings: list[str] = []
    timings: dict[str, float] = {}
    started = perf_counter()

    tick = perf_counter()
    try:
        caption = models.captioner.caption(image, beam=max(1, min(int(beam), 5)))
    except Exception as exc:
        caption = ""
        warnings.append(f"Captioning failed: {exc}")
    timings["caption"] = _elapsed_ms(tick)

    tick = perf_counter()
    segmentation = models.segmenter.predict(image)
    timings["segment"] = _elapsed_ms(tick)

    detections = ()
    if models.detector is not None:
        tick = perf_counter()
        try:
            detections = tuple(models.detector.detect(image))
        except Exception as exc:
            warnings.append(f"Detection failed: {exc}")
        timings["detect"] = _elapsed_ms(tick)

    mask = align_mask(segmentation.mask, image.size)
    segments = segment_stats(mask, segmentation.labels)
    dominant = dominant_label(segments)

    tick = perf_counter()
    overlay_png, mask_png = render_pngs(
        image, mask, segmentation.labels, alpha=overlay_alpha, detections=detections
    )
    timings["render"] = _elapsed_ms(tick)

    translated: str | None = None
    if lang != "en":
        if models.translator is None:
            warnings.append("Translation is disabled in this configuration.")
        elif not caption:
            warnings.append("Translation skipped because no caption was produced.")
        else:
            tick = perf_counter()
            try:
                translated = models.translator.translate(caption, lang)
            except Exception as exc:
                warnings.append(f"Translation to {lang} failed: {exc}")
            timings["translate"] = _elapsed_ms(tick)

    timings["total"] = _elapsed_ms(started)

    translator_version = "not used"
    if lang != "en" and models.translator is not None:
        translator_version = str(getattr(models.translator, "version", "unknown"))

    return AnalysisResult(
        caption=caption,
        segments=segments,
        dominant_label=dominant,
        image_size=image.size,
        overlay_png=overlay_png,
        mask_png=mask_png,
        translated=translated,
        target_lang=lang if lang != "en" else None,
        detections=detections,
        timings_ms=timings,
        model_versions={
            "segmenter": segmentation.version,
            "captioner": str(getattr(models.captioner, "version", models.captioner.name)),
            "translator": translator_version,
            "detector": str(getattr(models.detector, "version", "none")) if models.detector else "none",
        },
        warnings=tuple(warnings),
    )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0
