"""Stage 5, CHUNK: token-aware splitting that keeps the heading path.

With an 8k-token input limit almost every provision fits in one chunk, so splitting serves
retrieval quality, not the model (specification 8.2). The rules:

1. A body within `target_tokens` is one chunk.
2. Otherwise split on markdown headings, tracking the heading stack so each chunk knows its
   place ("Chapter II > § 4"), then pack paragraphs of a section up to the target.
3. A paragraph larger than `max_tokens` is hard-split on sentences, then on words, so nothing
   ever exceeds the model's limit.

`CHUNKER_VERSION` is recorded in the index; bump it when these rules change.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .parse import Document
from .tokenizer import Tokenizer

CHUNKER_VERSION = "1"

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+(?=\S)")


@dataclass(slots=True)
class Chunk:
    ordinal: int
    text: str
    token_count: int
    heading_path: str | None = None


def embedding_input(doc: Document, chunk: Chunk, prefix: str = "") -> str:
    """What is sent to the model: the title and heading path give a bare article its context."""
    head = doc.title
    if chunk.heading_path and chunk.heading_path != doc.title:
        head = f"{doc.title} > {chunk.heading_path}"
    return f"{prefix}{head}\n\n{chunk.text}"


@dataclass(slots=True)
class _Section:
    heading_path: str | None
    paragraphs: list[str]


def _sections(body: str) -> Iterator[_Section]:
    """Split the body on headings, keeping a heading stack per level."""
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> Iterator[_Section]:
        paragraphs = _paragraphs("\n".join(buf))
        buf.clear()
        if paragraphs:
            yield _Section(" > ".join(h for _, h in stack) or None, paragraphs)

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m and not line.startswith("    "):
            yield from flush()
            level, text = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            continue
        buf.append(line)
    yield from flush()


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _pack(paragraphs: list[str], tok: Tokenizer, target: int, max_tokens: int) -> Iterator[str]:
    """Pack consecutive paragraphs up to `target`; oversize ones are hard-split."""
    cur: list[str] = []
    cur_tokens = 0
    for p in paragraphs:
        n = tok.count(p)
        if n > max_tokens:
            if cur:
                yield "\n\n".join(cur)
                cur, cur_tokens = [], 0
            yield from _hard_split(p, tok, target, max_tokens)
            continue
        if cur and cur_tokens + n > target:
            yield "\n\n".join(cur)
            cur, cur_tokens = [], 0
        cur.append(p)
        cur_tokens += n
    if cur:
        yield "\n\n".join(cur)


def _hard_split(text: str, tok: Tokenizer, target: int, max_tokens: int) -> Iterator[str]:
    """Split an oversize paragraph on sentence ends, then on words, never above max_tokens."""
    pieces = _SENTENCE_END.split(text)
    cur: list[str] = []
    cur_tokens = 0
    for piece in pieces:
        n = tok.count(piece)
        if n > max_tokens:
            if cur:
                yield " ".join(cur)
                cur, cur_tokens = [], 0
            yield from _split_words(piece, tok, target)
            continue
        if cur and cur_tokens + n > target:
            yield " ".join(cur)
            cur, cur_tokens = [], 0
        cur.append(piece)
        cur_tokens += n
    if cur:
        yield " ".join(cur)


def _split_words(text: str, tok: Tokenizer, target: int) -> Iterator[str]:
    words = text.split()
    cur: list[str] = []
    for w in words:
        cur.append(w)
        if tok.count(" ".join(cur)) >= target:
            yield " ".join(cur)
            cur = []
    if cur:
        yield " ".join(cur)


def chunk_document(
    doc: Document, tok: Tokenizer, target_tokens: int = 1024, max_tokens: int = 8192
) -> list[Chunk]:
    """Split a document's body into chunks under the rules above. An empty body gives no chunks."""
    body = doc.body.strip()
    if not body:
        return []
    total = tok.count(body)
    if total <= target_tokens:
        return [Chunk(0, body, total, None)]
    chunks: list[Chunk] = []
    for section in _sections(body):
        for text in _pack(section.paragraphs, tok, target_tokens, max_tokens):
            chunks.append(Chunk(len(chunks), text, tok.count(text), section.heading_path))
    return chunks
