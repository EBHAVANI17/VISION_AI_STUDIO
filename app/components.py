"""Reusable Streamlit widgets for the Vision AI Studio app."""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

from vision_ai_studio.models.translator import LANGUAGES
from vision_ai_studio.schemas import AnalysisResult, Segment
from vision_ai_studio.viz.palette import hex_color

GRADIENT = "linear-gradient(120deg, #0f766e 0%, #1d4ed8 52%, #6d28d9 100%)"


def page_header() -> None:
    st.markdown(
        f"""
        <div class="vas-header" style="background:{GRADIENT}">
          <div class="vas-header-title">Vision AI Studio</div>
          <div class="vas-header-sub">
            Semantic segmentation · captioning · translation · speech
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state() -> None:
    st.markdown(
        """
        <div class="vas-empty">
          <div class="vas-empty-title">Upload an image to start</div>
          <div class="vas-empty-body">
            You will get a segmentation overlay, a ranked class breakdown, a natural-language
            caption, a translation and spoken audio. Everything runs on this machine.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def legend(segments: Sequence[Segment], limit: int = 8) -> None:
    if not segments:
        st.caption("No segments detected.")
        return
    rows = []
    for segment in segments[:limit]:
        color = hex_color(segment.class_id, segment.label)
        rows.append(
            '<div class="vas-legend-row">'
            f'<span class="vas-swatch" style="background:{color}"></span>'
            f'<span class="vas-legend-label">{html.escape(segment.label)}</span>'
            f'<span class="vas-legend-value">{segment.coverage:.1%}</span>'
            "</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def warnings(result: AnalysisResult) -> None:
    for message in result.warnings:
        st.warning(message, icon="⚠️")


def summary_metrics(result: AnalysisResult) -> None:
    first, second, third, fourth = st.columns(4)
    first.metric("Dominant class", result.dominant_label)
    second.metric("Classes detected", len(result.segments))
    third.metric("Analysis time", f"{result.timings_ms.get('total', 0) / 1000:.1f} s")
    width, height = result.image_size
    fourth.metric("Image", f"{width}×{height}")


def class_breakdown(segments: Sequence[Segment], limit: int) -> None:
    if not segments:
        st.info("No classes were detected.")
        return
    rows = [
        {
            "Class": segment.label,
            "Pixels": f"{segment.pixels:,}",
            "Coverage": f"{segment.coverage:.1%}",
        }
        for segment in segments
    ]
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("##### Largest classes")
    top = list(segments[:limit])
    largest = max((segment.coverage for segment in top), default=1.0) or 1.0
    for segment in top:
        st.progress(
            min(1.0, segment.coverage / largest),
            text=f"{segment.label} · {segment.coverage:.1%} of the image",
        )


def result_caption(result: AnalysisResult, target_lang: str) -> None:
    st.markdown("#### Caption")
    if result.caption:
        st.code(result.caption, language=None, wrap_lines=True)
    else:
        st.caption("_No caption was produced._")

    language = LANGUAGES.get(target_lang, target_lang)
    st.markdown(f"#### Translation · {html.escape(language)}")
    if target_lang == "en":
        st.caption("Choose another language in the sidebar to translate the caption.")
    elif result.translated:
        st.code(result.translated, language=None, wrap_lines=True)
    else:
        st.caption("_Translation unavailable._")


def download_buttons(result: AnalysisResult, *, prefix: str = "vision_ai") -> None:
    columns = st.columns(3)
    columns[0].download_button(
        "Download overlay PNG",
        data=result.overlay_png,
        file_name=f"{prefix}_overlay.png",
        mime="image/png",
        width="stretch",
        key=f"{prefix}_overlay",
    )
    columns[1].download_button(
        "Download mask PNG",
        data=result.mask_png,
        file_name=f"{prefix}_mask.png",
        mime="image/png",
        width="stretch",
        key=f"{prefix}_mask",
    )
    columns[2].download_button(
        "Download JSON",
        data=result.to_json().encode("utf-8"),
        file_name=f"{prefix}_result.json",
        mime="application/json",
        width="stretch",
        key=f"{prefix}_json",
    )


def model_details(result: AnalysisResult) -> None:
    with st.expander("Models & timings"):
        for name, version in result.model_versions.items():
            st.markdown(f"- **{name}**: `{version}`")
        timings = " · ".join(f"{name} {value:.0f} ms" for name, value in result.timings_ms.items())
        st.caption(timings)
