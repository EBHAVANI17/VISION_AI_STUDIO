"""Vision AI Studio — Streamlit front end."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from app import components
from vision_ai_studio import __version__
from vision_ai_studio.config import CAPTIONERS, SEGMENTERS, Settings
from vision_ai_studio.image import ImageError, image_digest, load_image
from vision_ai_studio.models.translator import LANGUAGES
from vision_ai_studio.pipeline import Models, analyze, build_models
from vision_ai_studio.schemas import AnalysisResult
from vision_ai_studio.tts import synthesize_mp3

st.set_page_config(
    page_title="Vision AI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLES = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1200px; }
  .vas-header {
    border-radius: 22px; padding: 2rem; color: #fff; margin-bottom: 1.4rem;
    box-shadow: 0 18px 40px -22px rgba(29, 78, 216, 0.65);
  }
  .vas-header-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; }
  .vas-header-sub { opacity: 0.92; margin-top: 0.35rem; font-size: 1rem; }
  .vas-empty {
    border: 1px dashed rgba(128, 128, 148, 0.5); border-radius: 20px;
    padding: 2.4rem 2rem; text-align: center; margin-top: 1rem;
  }
  .vas-empty-title { font-size: 1.2rem; font-weight: 650; margin-bottom: 0.5rem; }
  .vas-empty-body { opacity: 0.78; max-width: 46rem; margin: 0 auto; line-height: 1.5; }
  .vas-legend-row {
    display: flex; align-items: center; gap: 0.6rem; padding: 0.34rem 0;
    border-bottom: 1px solid rgba(128, 128, 148, 0.16);
  }
  .vas-swatch { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  .vas-legend-label { flex: 1; text-transform: capitalize; }
  .vas-legend-value { font-variant-numeric: tabular-nums; opacity: 0.75; }
</style>
"""


@st.cache_resource(show_spinner="Loading models (first run only)…")
def load_models(settings: Settings) -> Models:
    return build_models(settings)


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(
    image_bytes: bytes,
    settings: Settings,
    target_lang: str,
    beam: int,
    alpha: float,
) -> AnalysisResult:
    image = load_image(
        image_bytes,
        max_upload_mb=settings.max_upload_mb,
        max_edge=settings.max_edge,
    )
    return analyze(
        image,
        models=load_models(settings),
        target_lang=target_lang,
        overlay_alpha=alpha,
        beam=beam,
    )


def sidebar_controls(base: Settings) -> tuple[Settings, str, int, int]:
    with st.sidebar:
        st.markdown("### Analysis settings")
        segmenter = st.selectbox(
            "Segmentation model",
            options=list(SEGMENTERS),
            index=list(SEGMENTERS).index(base.segmenter),
            format_func=lambda key: SEGMENTERS[key],
        )
        captioner = st.selectbox(
            "Captioning model",
            options=list(CAPTIONERS),
            index=list(CAPTIONERS).index(base.captioner),
            format_func=lambda key: CAPTIONERS[key],
        )
        language = st.selectbox(
            "Caption language",
            options=list(LANGUAGES),
            format_func=lambda key: LANGUAGES[key],
        )
        quality = st.radio(
            "Caption search",
            options=["Fast", "Beam search"],
            horizontal=True,
            help="Beam search is slower but usually more accurate.",
        )
        alpha = st.slider(
            "Overlay opacity",
            min_value=0.2,
            max_value=0.8,
            value=float(base.overlay_alpha),
            step=0.05,
        )
        top_k = st.slider("Classes shown", min_value=3, max_value=15, value=8)

        st.divider()
        st.caption(
            "Models load once per session. Translation uses NLLB-200 (~2.4 GB) and is "
            "downloaded the first time you pick a non-English language."
        )
        if base.models == "fake":
            st.info("Demo mode: deterministic fake models, no weights required.")

    settings = base.with_overrides(segmenter=segmenter, captioner=captioner, overlay_alpha=alpha)
    return settings, language, 3 if quality == "Beam search" else 1, top_k


def audio_panel(result: AnalysisResult, settings: Settings) -> None:
    st.markdown("#### Speech")
    if settings.models == "fake":
        st.caption("Speech synthesis is disabled while using fake models.")
        return
    text = result.translated or result.caption
    if not text:
        st.caption("Nothing to speak.")
        return

    lang = result.target_lang or "en"
    cache_key = f"audio::{lang}::{abs(hash(text))}"
    if st.button("Generate speech", icon="🔊"):
        try:
            with st.spinner("Synthesizing speech…"):
                st.session_state[cache_key] = synthesize_mp3(text, lang)
        except Exception as exc:
            st.error(f"Speech synthesis failed: {exc}")

    audio = st.session_state.get(cache_key)
    if audio:
        st.audio(audio, format="audio/mp3")
        st.download_button(
            "Download MP3",
            data=audio,
            file_name="caption.mp3",
            mime="audio/mpeg",
            key=f"download-{cache_key}",
        )


def render_results(
    result: AnalysisResult,
    *,
    top_k: int,
    language: str,
    settings: Settings,
) -> None:
    components.warnings(result)
    components.summary_metrics(result)

    overview, segmentation, caption_tab = st.tabs(["Overview", "Segmentation", "Caption & speech"])
    with overview:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.image(
                result.overlay_png,
                caption="Segmentation overlay",
                width="stretch",
            )
        with right:
            st.markdown("##### Dominant class")
            st.markdown(f"## {result.dominant_label}")
            st.markdown("##### Class breakdown")
            components.legend(result.segments, limit=top_k)
            if result.caption:
                st.markdown("##### Caption")
                st.markdown(f"**{result.caption}**")

    with segmentation:
        st.image(result.mask_png, caption="Colorized mask", width="stretch")
        components.class_breakdown(result.segments, top_k)
        components.download_buttons(result)

    with caption_tab:
        components.result_caption(result, language)
        audio_panel(result, settings)
        st.divider()
        components.model_details(result)


def main() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    components.page_header()

    try:
        base = Settings.from_env()
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")
        return

    settings, language, beam, top_k = sidebar_controls(base)

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help=f"Up to {settings.max_upload_mb} MB. Images are processed in memory and not stored.",
    )
    if uploaded is None:
        components.empty_state()
        return

    raw = uploaded.getvalue()
    try:
        image = load_image(
            raw,
            max_upload_mb=settings.max_upload_mb,
            max_edge=settings.max_edge,
        )
    except ImageError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(image, caption=f"{image.width}×{image.height}", width="stretch")
    with right:
        st.markdown("### Ready when you are")
        st.markdown(f"**Segmentation** · {SEGMENTERS[settings.segmenter]}")
        st.markdown(f"**Captioning** · {CAPTIONERS[settings.captioner]}")
        if language != "en":
            st.markdown(f"**Translation** · {LANGUAGES[language]}")
        analyze_clicked = st.button("Analyze image", type="primary", width="stretch")

    if analyze_clicked:
        spinner_text = "Segmenting and captioning"
        spinner_text += " + translating" if language != "en" else ""
        try:
            with st.spinner(f"{spinner_text}…"):
                result = run_analysis(raw, settings, language, beam, settings.overlay_alpha)
            st.session_state["vas_result"] = result
            st.session_state["vas_meta"] = {
                "digest": image_digest(raw),
                "settings": settings.cache_key(),
                "language": language,
                "beam": beam,
            }
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    result = st.session_state.get("vas_result")
    if result is None:
        return

    meta = st.session_state.get("vas_meta", {})
    if meta.get("digest") != image_digest(raw):
        st.info("Showing the previous image's results. Press Analyze to run on this image.")
    elif meta.get("settings") != settings.cache_key() or meta.get("language") != language:
        st.info("Settings changed. Press Analyze to refresh the results.")

    st.divider()
    render_results(result, top_k=top_k, language=language, settings=settings)


main()

st.sidebar.caption(f"Vision AI Studio v{__version__} · CPU inference, no image retention")
