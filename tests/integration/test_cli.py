from __future__ import annotations

import json
import sys
from pathlib import Path

from vision_ai_studio import cli
from vision_ai_studio.cli import main


def test_info_runs_in_fake_mode(capsys) -> None:
    assert main(["info", "--models", "fake"]) == 0
    output = capsys.readouterr().out
    assert "vision-ai-studio" in output
    assert "cpu (fake mode)" in output
    assert "segmenter" in output


class _Stream:
    def __init__(self) -> None:
        self.kwargs: dict[str, str] | None = None

    def reconfigure(self, **kwargs: str) -> None:
        self.kwargs = kwargs


def test_configure_stdio_never_raises(monkeypatch) -> None:
    stream = _Stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", object())
    cli._configure_stdio()
    assert stream.kwargs == {"errors": "replace"}


def test_analyze_writes_artifacts(scene_path: Path, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "out"
    code = main(
        [
            "analyze",
            str(scene_path),
            "--models",
            "fake",
            "--translator",
            "none",
            "-o",
            str(outdir),
        ]
    )
    assert code == 0

    payload = json.loads((outdir / "scene.json").read_text(encoding="utf-8"))
    assert payload["caption"].startswith("a synthetic test scene")
    assert payload["dominant_label"] == "person"
    assert (outdir / "scene.overlay.png").read_bytes().startswith(b"\x89PNG")
    assert (outdir / "scene.mask.png").read_bytes().startswith(b"\x89PNG")

    output = capsys.readouterr().out
    assert "Caption:" in output
    assert "person" in output


def test_analyze_translates_in_fake_mode(scene_path: Path, tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    code = main(
        [
            "analyze",
            str(scene_path),
            "--models",
            "fake",
            "--translator",
            "nllb",
            "--lang",
            "hi",
            "-o",
            str(outdir),
        ]
    )
    assert code == 0
    payload = json.loads((outdir / "scene.json").read_text(encoding="utf-8"))
    assert payload["target_lang"] == "hi"
    assert payload["translated"].startswith("[hi]")


def test_analyze_quiet_prints_only_caption(scene_path: Path, tmp_path: Path, capsys) -> None:
    code = main(
        [
            "analyze",
            str(scene_path),
            "--models",
            "fake",
            "--translator",
            "none",
            "--quiet",
            "-o",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == (
        "a synthetic test scene with a person next to a chair"
    )


def test_analyze_missing_file_fails(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "analyze",
            str(tmp_path / "missing.png"),
            "--models",
            "fake",
            "-o",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "File not found" in capsys.readouterr().err
