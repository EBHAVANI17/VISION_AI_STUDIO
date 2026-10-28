from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


class _Upload:
    """Minimal stand-in for ``st.file_uploader``'s UploadedFile."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("VAS_MODELS", "fake")
    instance = AppTest.from_file(str(APP_PATH), default_timeout=30)
    instance.run()
    return instance


@pytest.fixture
def analyzed(monkeypatch: pytest.MonkeyPatch, scene_png: bytes) -> AppTest:
    monkeypatch.setenv("VAS_MODELS", "fake")
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: _Upload(scene_png))
    instance = AppTest.from_file(str(APP_PATH), default_timeout=30)
    instance.run()
    instance.button[0].click().run()
    return instance


def test_app_boots_without_errors(app: AppTest) -> None:
    assert not app.exception
    assert any("Vision AI Studio" in markdown.value for markdown in app.markdown)
    assert any("Upload an image to start" in markdown.value for markdown in app.markdown)


def test_sidebar_exposes_settings(app: AppTest) -> None:
    labels = [selectbox.label for selectbox in app.selectbox]
    assert "Segmentation model" in labels
    assert "Captioning model" in labels
    assert "Caption language" in labels
    assert [slider.label for slider in app.slider] == ["Overlay opacity", "Classes shown"]
    assert any("Demo mode" in info.value for info in app.info)


def test_uploader_accepts_expected_types(app: AppTest) -> None:
    assert len(app.get("file_uploader")) == 1
    uploader = app.get("file_uploader")[0]
    assert [item.lstrip(".") for item in uploader.proto.type] == ["jpg", "jpeg", "png", "webp"]


def test_analyze_flow_renders_results(analyzed: AppTest) -> None:
    assert not analyzed.exception
    metrics = {metric.label: metric.value for metric in analyzed.metric}
    assert metrics["Dominant class"] == "person"
    assert metrics["Classes detected"] == "3"
    assert metrics["Image"].startswith("320")
    assert metrics["Image"].endswith("240")

    captions = [code.value for code in analyzed.code]
    assert captions[0] == "a synthetic test scene with a person next to a chair"
    assert len(analyzed.get("download_button")) == 3

    legend = " ".join(markdown.value for markdown in analyzed.markdown)
    assert "person" in legend
    assert "chair" in legend


def test_translation_language_reaches_the_result(
    monkeypatch: pytest.MonkeyPatch, scene_png: bytes
) -> None:
    monkeypatch.setenv("VAS_MODELS", "fake")
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: _Upload(scene_png))
    instance = AppTest.from_file(str(APP_PATH), default_timeout=30)
    instance.run()
    instance.selectbox[2].select("hi").run()
    instance.button[0].click().run()

    assert not instance.exception
    assert "[hi] a synthetic test scene" in " ".join(code.value for code in instance.code)
