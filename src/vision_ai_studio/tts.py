"""Text-to-speech with gTTS, cached by (text, language)."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

from vision_ai_studio.models.translator import LANGUAGES, normalize_lang

TTS_CODES: dict[str, str] = {"zh": "zh-CN"}


class GttsSynthesizer:
    name = "gtts"
    version = "gtts"

    def speak(self, text: str, lang: str) -> bytes:
        return synthesize_mp3(text, lang)


class NullSynthesizer:
    name = "none"
    version = "none"

    def speak(self, text: str, lang: str) -> bytes:
        del text, lang
        return b""


@lru_cache(maxsize=128)
def synthesize_mp3(text: str, lang: str) -> bytes:
    code = normalize_lang(lang)
    if code not in LANGUAGES:
        raise ValueError(f"No speech voice for {lang!r}")
    if not text.strip():
        return b""
    try:
        from gtts import gTTS
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError("Install `gtts` to enable speech synthesis.") from exc

    buffer = BytesIO()
    gTTS(text=text, lang=TTS_CODES.get(code, code), slow=False).write_to_fp(buffer)
    return buffer.getvalue()


def build_synthesizer(name: str = "gtts") -> GttsSynthesizer | NullSynthesizer:
    if name == "none":
        return NullSynthesizer()
    if name == "gtts":
        return GttsSynthesizer()
    raise ValueError(f"Unknown synthesizer {name!r}")
