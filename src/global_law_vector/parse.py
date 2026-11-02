"""Stage 4, PARSE: one provision file into a Document.

The corpora differ in shape (specification section 4): some files open with YAML front matter,
some with a `#` heading, some are bare text whose identity lives only in the path. The parser
tolerates all of them and never fails on a law file: at worst the title comes from the path.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_KV = re.compile(r"^([A-Za-z0-9_\-.]+)\s*:\s*(.*)$")
_CITATION_KEYS = ("citation", "citace", "cite", "citacao", "citação", "zitat", "ref")


@dataclass(slots=True)
class Document:
    path: str  # relative to the global-law root, forward slashes
    corpus_path: str
    title: str
    body: str  # markdown body without front matter
    front_matter: dict[str, Any] = field(default_factory=dict)
    citation: str | None = None
    content_sha: str = ""
    byte_size: int = 0

    @property
    def relative_in_corpus(self) -> str:
        return self.path[len(self.corpus_path) :].lstrip("/")


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a leading `---` block into a dict and return (dict, rest).

    The corpora use flat `key: value` front matter (Czechia: citace, nazev, druh, eli). This
    reads that subset of YAML: scalars, quoted strings, and simple `[a, b]` lists. Anything
    nested is kept as its raw text. No YAML library is needed.
    """
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    data: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        kv = _KV.match(line)
        if not kv:
            continue
        key, raw = kv.group(1), kv.group(2).strip()
        data[key] = _scalar(raw)
    return data, text[m.end() :]


def _scalar(raw: str) -> Any:
    if raw == "":
        return ""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        inner = raw[1:-1]
        return inner.replace('\\"', '"') if raw[0] == '"' else inner
    if raw.startswith("[") and raw.endswith("]"):
        return [_scalar(p.strip()) for p in raw[1:-1].split(",") if p.strip()]
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def first_heading(body: str) -> str | None:
    """The first `#` heading; failing that, the first heading of any level."""
    fallback: str | None = None
    for line in body.splitlines():
        m = _HEADING.match(line)
        if not m:
            continue
        if len(m.group(1)) == 1:
            return m.group(2)
        if fallback is None:
            fallback = m.group(2)
    return fallback


def title_from_path(path: str, corpus_path: str) -> str:
    """ "lei-1990-07004 / art-0001" for a bare file, from its path inside the corpus."""
    rel = PurePosixPath(path[len(corpus_path) :].lstrip("/"))
    parts = list(rel.with_suffix("").parts)
    return " / ".join(parts) if parts else path


def parse_document(path: str, corpus_path: str, raw: bytes) -> Document:
    """Parse a provision file. `path` is relative to the root; `raw` is the file's bytes."""
    text = raw.decode("utf-8", errors="replace")
    if text.startswith("\ufeff"):  # byte-order mark
        text = text[1:]
    front, body = parse_front_matter(text)
    body = body.strip("\n")
    title = first_heading(body)
    if not title:
        for key in ("title", "nazev", "name", "titulo", "título"):
            value = front.get(key)
            if isinstance(value, str) and value.strip():
                title = value.strip()
                break
    if not title:
        title = title_from_path(path, corpus_path)
    citation = None
    for key in _CITATION_KEYS:
        value = front.get(key)
        if isinstance(value, str) and value.strip():
            citation = value.strip()
            break
    return Document(
        path=path,
        corpus_path=corpus_path,
        title=title,
        body=body,
        front_matter=front,
        citation=citation,
        content_sha=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
    )


def content_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
