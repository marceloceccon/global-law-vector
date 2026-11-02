"""Stage 4, PARSE: one provision file into a Document.

The corpora differ in shape (specification section 4): some files open with YAML front matter,
some with a `#` heading, some are bare text whose identity lives only in the path. The parser
tolerates all of them and never fails on a law file: at worst the title comes from the path.
"""

from __future__ import annotations

import hashlib
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
log = logging.getLogger(__name__)

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

    The corpora use flat `key: value` front matter (Czechia: citace, nazev, druh, eli); a
    survey of the checked-out corpora on 2026-11-02 found nothing else. This reads that subset
    of YAML without a YAML library: scalars, quoted strings, `[a, b]` lists, block lists
    (`- item` lines under a key), and block scalars (`|` keeps line breaks, `>` folds them).
    Anything else nested under a key is kept as its raw, dedented text.
    """
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    data: dict[str, Any] = {}
    lines = m.group(1).splitlines()
    i = 0
    while i < len(lines):
        kv = _KV.match(lines[i])
        i += 1
        if not kv:
            continue
        block: list[str] = []
        while i < len(lines) and (
            lines[i][:1] in (" ", "\t") or lines[i].startswith("- ") or not lines[i].strip()
        ):
            block.append(lines[i])
            i += 1
        while block and not block[-1].strip():
            block.pop()
        data[kv.group(1)] = _value(kv.group(2).strip(), block)
    return data, text[m.end() :]


def _value(raw: str, block: list[str]) -> Any:
    """A key's value from the rest of its line and the indented lines under it."""
    if raw[:1] in ("|", ">") and raw.rstrip("+-") in ("|", ">"):
        body = textwrap.dedent("\n".join(block))
        if raw.startswith(">"):  # folded: single line breaks become spaces
            body = "\n".join(" ".join(p.split("\n")) for p in body.split("\n\n"))
        return body + "\n" if raw.endswith("+") else body
    if not block:
        return _scalar(raw)
    items = [line.strip() for line in block if line.strip()]
    if raw == "" and all(item.startswith("- ") or item == "-" for item in items):
        return [_scalar(item[1:].strip()) for item in items]
    if raw == "":
        return textwrap.dedent("\n".join(block))  # a nested mapping, kept as text
    return " ".join([raw, *items])  # a plain scalar continued on the next lines


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
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
        log.warning(
            "%s is not valid UTF-8; %d undecodable sequences replaced with U+FFFD",
            path,
            text.count("\ufffd"),
        )
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
