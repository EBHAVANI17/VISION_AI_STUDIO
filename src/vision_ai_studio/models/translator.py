"""Caption translation backends (NLLB by default, Google as opt-in fallback)."""

from __future__ import annotations

from vision_ai_studio.models.base import ModelLoadError

LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese (Simplified)",
}

NLLB_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
}

GOOGLE_CODES: dict[str, str] = {
    "zh": "zh-CN",
}

NLLB_REPO = "facebook/nllb-200-distilled-600M"


def normalize_lang(code: str) -> str:
    base = code.strip().lower().replace("_", "-").split("-")[0]
    if base not in LANGUAGES:
        raise ValueError(f"Unsupported language {code!r}; choose one of {sorted(LANGUAGES)}")
    return base


class NullTranslator:
    name = "none"
    version = "none"

    def translate(self, text: str, target: str) -> str:
        normalize_lang(target)
        return text


class NllbTranslator:
    """Offline seq2seq translation from English into the configured languages."""

    name = "nllb"

    def __init__(
        self, *, device: str = "cpu", revision: str | None = None, repo_id: str = NLLB_REPO
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ModelLoadError("transformers and torch are required for translation.") from exc

        self._torch = torch
        self._device = torch.device(device)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                repo_id, revision=revision, src_lang=NLLB_CODES["en"]
            )
            self._model = (
                AutoModelForSeq2SeqLM.from_pretrained(repo_id, revision=revision)
                .to(self._device)
                .eval()
            )
        except Exception as exc:
            raise ModelLoadError(f"Could not load {repo_id}: {exc}") from exc

        commit = getattr(self._model.config, "_commit_hash", None) or revision or "main"
        self.version = f"hf:{repo_id}@{commit}"

    def translate(self, text: str, target: str) -> str:
        lang = normalize_lang(target)
        if lang == "en" or not text.strip():
            return text
        torch = self._torch
        encoded = self._tokenizer(text, src_lang=NLLB_CODES["en"], return_tensors="pt").to(
            self._device
        )
        forced_bos = self._tokenizer.convert_tokens_to_ids(NLLB_CODES[lang])
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_new_tokens=128,
                num_beams=2,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return str(decoded[0]).strip()


class GoogleTranslator:
    """Web-backed fallback; unofficial and rate-limited, kept behind an explicit opt-in."""

    name = "google"
    version = "deep-translator:google"

    def translate(self, text: str, target: str) -> str:
        lang = normalize_lang(target)
        if lang == "en" or not text.strip():
            return text
        try:
            from deep_translator import GoogleTranslator as _Google
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError("Install `deep-translator` to use the Google backend.") from exc
        translated = _Google(source="auto", target=GOOGLE_CODES.get(lang, lang)).translate(text)
        return str(translated)


def build_translator(
    name: str, *, device: str = "cpu", revision: str | None = None
) -> NullTranslator | NllbTranslator | GoogleTranslator:
    if name == "none":
        return NullTranslator()
    if name == "nllb":
        return NllbTranslator(device=device, revision=revision)
    if name == "google":
        return GoogleTranslator()
    raise ValueError(f"Unknown translator {name!r}")
