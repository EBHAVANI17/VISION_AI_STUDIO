"""FastAPI backend service for Vision AI Studio."""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image

from vision_ai_studio import __version__
from vision_ai_studio.config import CAPTIONERS, SEGMENTERS, Settings
from vision_ai_studio.image import ImageError, load_image
from vision_ai_studio.models.detector import build_detector
from vision_ai_studio.models.translator import LANGUAGES
from vision_ai_studio.pipeline import Models, analyze, build_models

# Runtime Prometheus metrics counters
METRICS = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "last_latency_ms": 0.0,
}

_models_cache: dict[str, Models] = {}


def get_cached_models(settings: Settings) -> Models:
    key = f"{settings.models}:{settings.segmenter}:{settings.captioner}:{settings.translator}"
    if key not in _models_cache:
        _models_cache[key] = build_models(settings)
    return _models_cache[key]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Warm up default fake models if in testing or default config
    settings = Settings.from_env()
    get_cached_models(settings)
    yield
    _models_cache.clear()


app = FastAPI(
    title="Vision AI Studio API",
    description="Production REST API for image segmentation, captioning, translation and YOLO detection.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> dict[str, Any]:
    """Healthcheck endpoint for orchestrators (Kubernetes, Render, Cloud Run)."""
    settings = Settings.from_env()
    return {
        "status": "healthy",
        "version": __version__,
        "models_mode": settings.models,
        "device": settings.device,
        "timestamp": time.time(),
    }


@app.get("/info", status_code=status.HTTP_200_OK)
def info() -> dict[str, Any]:
    """Inspect model registry options and supported features."""
    settings = Settings.from_env()
    return {
        "version": __version__,
        "available_segmenters": list(SEGMENTERS.keys()),
        "available_captioners": list(CAPTIONERS.keys()),
        "supported_languages": LANGUAGES,
        "active_settings": asdict(settings),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus-compatible plain text metrics."""
    return (
        f"# HELP vas_requests_total Total number of analysis requests processed\n"
        f"# TYPE vas_requests_total counter\n"
        f"vas_requests_total {METRICS['requests_total']}\n"
        f"# HELP vas_requests_success Total number of successful requests\n"
        f"# TYPE vas_requests_success counter\n"
        f"vas_requests_success {METRICS['requests_success']}\n"
        f"# HELP vas_requests_error Total number of failed requests\n"
        f"# TYPE vas_requests_error counter\n"
        f"vas_requests_error {METRICS['requests_error']}\n"
        f"# HELP vas_last_latency_ms Latency of the most recent request in milliseconds\n"
        f"# TYPE vas_last_latency_ms gauge\n"
        f"vas_last_latency_ms {METRICS['last_latency_ms']:.2f}\n"
    )


@app.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_image(
    file: UploadFile = File(...),
    segmenter: str = Query("ade", enum=["ade", "voc"]),
    captioner: str = Query("blip-base", enum=["blip-base", "blip-large"]),
    target_lang: str = Query("en"),
    enable_detector: bool = Query(False),
    overlay_alpha: float = Query(0.45, ge=0.0, le=1.0),
    beam: int = Query(1, ge=1, le=5),
) -> dict[str, Any]:
    """Run full multimodal inference pipeline on an uploaded image."""
    METRICS["requests_total"] += 1
    start = time.perf_counter()

    try:
        contents = await file.read()
        image = load_image(contents)
    except ImageError as exc:
        METRICS["requests_error"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {exc}",
        ) from exc
    except Exception as exc:
        METRICS["requests_error"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read image upload: {exc}",
        ) from exc

    try:
        settings = Settings.from_env()
        # Override with query parameters
        active_settings = Settings(
            segmenter=segmenter,
            captioner=captioner,
            translator=settings.translator,
            models=settings.models,
            device=settings.device,
        )
        base_models = get_cached_models(active_settings)

        detector = None
        if enable_detector:
            detector = build_detector(fake=(settings.models == "fake"))

        models = Models(
            segmenter=base_models.segmenter,
            captioner=base_models.captioner,
            translator=base_models.translator,
            detector=detector,
        )

        result = analyze(
            image,
            models=models,
            target_lang=target_lang,
            overlay_alpha=overlay_alpha,
            beam=beam,
        )

        latency = (time.perf_counter() - start) * 1000.0
        METRICS["requests_success"] += 1
        METRICS["last_latency_ms"] = latency

        return result.to_payload()
    except Exception as exc:
        METRICS["requests_error"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline processing failed: {exc}",
        ) from exc
