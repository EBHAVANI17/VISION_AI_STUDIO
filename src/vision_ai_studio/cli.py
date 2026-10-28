"""Command line interface: ``vas analyze``, ``vas prefetch``, ``vas info``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from contextlib import suppress
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

from vision_ai_studio import __version__
from vision_ai_studio.config import CAPTIONERS, MODES, SEGMENTERS, TRANSLATORS, Settings
from vision_ai_studio.image import ImageError, load_image
from vision_ai_studio.models.base import ModelInferenceError, ModelLoadError
from vision_ai_studio.models.registry import get_translator, resolve_device
from vision_ai_studio.models.translator import LANGUAGES
from vision_ai_studio.pipeline import analyze, build_models

log = logging.getLogger("vision_ai_studio")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


def _configure_stdio() -> None:
    """Never crash while printing translated text on a legacy Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with suppress(ValueError, OSError):
            reconfigure(errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vas",
        description="Vision AI Studio: segmentation, captioning, translation and speech.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    info = sub.add_parser("info", help="Show configuration, device and installed versions.")
    _add_settings_args(info)
    info.set_defaults(handler=_cmd_info)

    prefetch = sub.add_parser("prefetch", help="Download model weights ahead of time.")
    _add_settings_args(prefetch)
    prefetch.set_defaults(handler=_cmd_prefetch)

    run = sub.add_parser("analyze", help="Analyze a single image and write JSON + PNGs.")
    run.add_argument("image", type=Path, help="Path to a JPEG/PNG/WebP image.")
    run.add_argument("-o", "--outdir", type=Path, default=Path("outputs"))
    run.add_argument("--lang", default="en", choices=sorted(LANGUAGES), help="Caption language.")
    run.add_argument("--beam", type=int, default=1, help="Caption beam size (1-5).")
    run.add_argument("--alpha", type=float, default=None, help="Overlay opacity (0-1).")
    run.add_argument("--quiet", action="store_true", help="Only print the caption.")
    _add_settings_args(run)
    run.set_defaults(handler=_cmd_analyze)
    return parser


def _add_settings_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seg", choices=sorted(SEGMENTERS), help="Segmentation model.")
    parser.add_argument("--caption", choices=sorted(CAPTIONERS), help="Captioning model.")
    parser.add_argument("--translator", choices=sorted(TRANSLATORS), help="Translation backend.")
    parser.add_argument("--device", help="auto, cpu, cuda or mps.")
    parser.add_argument("--models", choices=sorted(MODES), help="Use real or fake models.")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides: dict[str, object] = {}
    if getattr(args, "seg", None):
        overrides["segmenter"] = args.seg
    if getattr(args, "caption", None):
        overrides["captioner"] = args.caption
    if getattr(args, "translator", None):
        overrides["translator"] = args.translator
    if getattr(args, "device", None):
        overrides["device"] = args.device
    if getattr(args, "models", None):
        overrides["models"] = args.models
    if getattr(args, "alpha", None) is not None:
        overrides["overlay_alpha"] = args.alpha
    return settings.with_overrides(**overrides)


def _cmd_analyze(args: argparse.Namespace) -> int:
    try:
        settings = _settings_from_args(args)
        image = load_image(
            Path(args.image),
            max_upload_mb=settings.max_upload_mb,
            max_edge=settings.max_edge,
        )
        result = analyze(
            image,
            models=build_models(settings),
            target_lang=args.lang,
            overlay_alpha=settings.overlay_alpha,
            beam=args.beam,
        )
    except (ImageError, ModelLoadError, ModelInferenceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:  # pragma: no cover - defensive, still logged with a traceback
        log.exception("Unexpected failure")
        return 1

    if args.quiet:
        print(result.caption)
        return 0

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    json_path = outdir / f"{stem}.json"
    overlay_path = outdir / f"{stem}.overlay.png"
    mask_path = outdir / f"{stem}.mask.png"
    json_path.write_text(result.to_json(), encoding="utf-8")
    overlay_path.write_bytes(result.overlay_png)
    mask_path.write_bytes(result.mask_png)

    width, height = result.image_size
    print(f"Image:      {width}x{height}")
    print(f"Caption:    {result.caption or '(none)'}")
    if result.translated:
        print(f"Translated: ({result.target_lang}) {result.translated}")
    print(f"Dominant:   {result.dominant_label}")
    top = " | ".join(f"{s.label} {s.coverage:.1%}" for s in result.segments[:5])
    print(f"Classes:    {top}")
    timings = " | ".join(f"{name} {value:.0f} ms" for name, value in result.timings_ms.items())
    print(f"Timings:    {timings}")
    for warning in result.warnings:
        print(f"Warning:    {warning}", file=sys.stderr)
    print(f"Wrote:      {json_path}, {overlay_path}, {mask_path}")
    return 0


def _cmd_prefetch(args: argparse.Namespace) -> int:
    try:
        settings = _settings_from_args(args)
        settings = settings.with_overrides(translator="none")
        models = build_models(settings)
        print(f"segmenter: {models.segmenter.name} -> {models.segmenter.version}")
        print(f"captioner: {models.captioner.name} -> {models.captioner.version}")
        if args.translator and args.translator != "none":
            device = "cpu" if settings.models == "fake" else resolve_device(settings.device)
            translator = get_translator(args.translator, device, None, settings.models)
            print(f"translator: {translator.name} -> {translator.version}")
    except (ModelLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:  # pragma: no cover
        log.exception("Unexpected failure")
        return 1
    print("prefetch complete")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    try:
        settings = _settings_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"vision-ai-studio {__version__}")
    print(f"  segmenter:   {settings.segmenter} ({SEGMENTERS[settings.segmenter]})")
    print(f"  captioner:   {settings.captioner} ({CAPTIONERS[settings.captioner]})")
    print(f"  translator:  {settings.translator} ({TRANSLATORS[settings.translator]})")
    print(f"  models:      {settings.models}")
    print(f"  max edge:    {settings.max_edge}px")
    print(f"  upload cap:  {settings.max_upload_mb} MB")
    print(f"  overlay:     {settings.overlay_alpha:.2f}")
    if settings.models == "fake":
        print("  device:      cpu (fake mode)")
    else:
        try:
            print(f"  device:      {resolve_device(settings.device)}")
        except ValueError as exc:
            print(f"  device:      error: {exc}")
    for package in ("torch", "torchvision", "transformers", "streamlit", "gtts"):
        try:
            print(f"  {package:<11} {pkg_version(package)}")
        except PackageNotFoundError:
            print(f"  {package:<11} not installed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
