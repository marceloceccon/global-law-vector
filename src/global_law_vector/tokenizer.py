"""Token counting for the chunker.

The exact tokenizer is the model's (Hugging Face `tokenizers`, loaded from a local file or the
hub). When it is not installed or cannot be fetched, a heuristic stands in. The index records
whether its token counts are exact or estimated and refuses to mix the two, because the
chunk boundaries differ.

A `tokenizer.json` carries the settings the model was trained with, including truncation (512
tokens for arctic-embed-m-v2.0) and padding. Both are switched off here: a counter that stops
at 512 would make every document look short enough to stay in one piece.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# Kana, CJK ideographs and Hangul syllables: roughly one token per character.
_CJK = re.compile("[\u3040-\u9fff\uac00-\ud7af]")


class Tokenizer(Protocol):
    name: str
    exact: bool

    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """About 1.3 subword tokens per word for Latin-script legal text, more for CJK."""

    name = "heuristic"
    exact = False

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = _WORD.findall(text)
        cjk = len(_CJK.findall(text))
        return int(len(words) * 1.3) + cjk + 2


class HFTokenizer:
    name: str
    exact = True

    def __init__(self, tok, name: str) -> None:
        tok.no_truncation()
        tok.no_padding()
        self._tok = tok
        self.name = name

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=True).ids)

    @classmethod
    def from_file(cls, path: Path, name: str | None = None) -> HFTokenizer:
        from tokenizers import Tokenizer as _T

        return cls(_T.from_file(str(path)), name or f"file:{path.name}")

    @classmethod
    def from_hub(cls, model: str) -> HFTokenizer:
        from tokenizers import Tokenizer as _T

        return cls(_T.from_pretrained(model), model)


def cached_tokenizer_path(cache_dir: Path, model: str) -> Path:
    return Path(cache_dir) / f"tokenizer-{model.replace('/', '--')}.json"


def load_tokenizer(
    model: str,
    path: Path | None = None,
    allow_network: bool = True,
    cache_dir: Path | None = None,
) -> Tokenizer:
    """The model's tokenizer when available, else the heuristic (with a warning).

    Order: an explicit file, then a copy cached under `cache_dir` from an earlier run, then the
    hub (saving the copy, so later runs make no network request), then the heuristic.
    """
    name = None
    if path is None and cache_dir is not None:
        cached = cached_tokenizer_path(cache_dir, model)
        if cached.is_file():
            path, name = cached, model  # the hub's tokenizer, saved by an earlier run
    if path is not None:
        try:
            return HFTokenizer.from_file(path, name)
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            log.warning("tokenizer file %s unusable (%s); using heuristic", path, exc)
            return HeuristicTokenizer()
    if allow_network:
        try:
            tok = HFTokenizer.from_hub(model)
            if cache_dir is not None:
                try:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    tok._tok.save(str(cached_tokenizer_path(cache_dir, model)))
                except Exception as exc:  # noqa: BLE001
                    log.warning("could not cache the tokenizer: %s", exc)
            return tok
        except ImportError:
            log.warning("`tokenizers` not installed; token counts are heuristic")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not fetch tokenizer for %s (%s); using heuristic", model, exc)
    return HeuristicTokenizer()
