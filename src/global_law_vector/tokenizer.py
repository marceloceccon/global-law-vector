"""Token counting for the chunker.

The exact tokenizer is the model's (Hugging Face `tokenizers`, loaded from a local file or the
hub). When it is not installed or cannot be fetched, a heuristic stands in: it over-estimates
slightly, which keeps chunks under the limit. The database records which one built the index.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """About 1.3 subword tokens per word for Latin-script legal text, more for CJK."""

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = _WORD.findall(text)
        cjk = sum(1 for ch in text if "぀" <= ch <= "鿿")
        return int(len(words) * 1.3) + cjk + 2


class HFTokenizer:
    name: str

    def __init__(self, tok, name: str) -> None:
        self._tok = tok
        self.name = name

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=True).ids)

    @classmethod
    def from_file(cls, path: Path) -> HFTokenizer:
        from tokenizers import Tokenizer as _T

        return cls(_T.from_file(str(path)), f"file:{path.name}")

    @classmethod
    def from_hub(cls, model: str) -> HFTokenizer:
        from tokenizers import Tokenizer as _T

        return cls(_T.from_pretrained(model), model)


def load_tokenizer(model: str, path: Path | None = None, allow_network: bool = True) -> Tokenizer:
    """The model's tokenizer when available, else the heuristic (with a warning)."""
    if path is not None:
        try:
            return HFTokenizer.from_file(path)
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            log.warning("tokenizer file %s unusable (%s); using heuristic", path, exc)
            return HeuristicTokenizer()
    if allow_network:
        try:
            return HFTokenizer.from_hub(model)
        except ImportError:
            log.warning("`tokenizers` not installed; token counts are heuristic")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not fetch tokenizer for %s (%s); using heuristic", model, exc)
    return HeuristicTokenizer()
