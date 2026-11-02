"""Stage 5, CHUNK: token-aware splitting that keeps the heading path.

The model accepts 8,192 tokens, so splitting serves retrieval quality rather than the model: a
search hit should land on the part of a long act that answers the question, not on a vector
averaged over the whole act (specification 8.2). The rules:

1. A body within `target_tokens` is one chunk, headings and all.
2. Otherwise the body is split on markdown headings, tracking the heading stack so each chunk
   knows its place ("Chapter II > § 4"), and the paragraphs of each section are packed into
   chunks of at most `target_tokens`.
3. A paragraph larger than the target is split on sentence ends and line breaks, then on
   words, then, for long runs without spaces such as Chinese text, on characters. The pieces
   keep their original spacing and line breaks.

Every piece of text is counted a constant number of times (plus a logarithmic factor where
words or characters are cut), so chunking stays linear in the size of the corpus.

No chunk exceeds `target_tokens` as counted by the tokenizer. Chunks do not overlap: statutes
are split on their own structure, and a provision cut in two keeps its heading path in both
halves. `CHUNKER_VERSION` is recorded in the index; bump it whenever these rules or the text
sent to the model change.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from .parse import Document
from .tokenizer import Tokenizer

# 2: token counts no longer stop at the tokenizer's 512-token truncation; paragraphs above the
#    target are split; spacing is kept when splitting; a text that opens with its own title
#    is not sent with the title twice.
CHUNKER_VERSION = "2"

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# A break after sentence punctuation followed by whitespace, after CJK sentence punctuation,
# or at a line break.
_BREAK = re.compile(r"(?<=[.;:!?])\s+|(?<=[\u3002\uff01\uff1f\uff1b])|\n")
_WORD = re.compile(r"\S+\s*")


@dataclass(slots=True)
class Chunk:
    ordinal: int
    text: str
    token_count: int
    heading_path: str | None = None


def embedding_input(doc: Document, chunk: Chunk, prefix: str = "") -> str:
    """What is sent to the model: the title and heading path give a bare article its context.

    A whole provision that already opens with its title as a heading is sent as it is.
    """
    if chunk.heading_path is None:
        first = _HEADING.match(chunk.text.split("\n", 1)[0])
        if first and first.group(2) == doc.title:
            return f"{prefix}{chunk.text}"
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
        if m:
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


def _pack(units: list[str], sep: str, tok: Tokenizer, target: int) -> Iterator[str]:
    """Join consecutive units with `sep` into texts of at most `target` tokens. A unit that is
    larger on its own is split further.

    Each unit is counted once and the counts are summed, less the special tokens the tokenizer
    adds to every text, so packing stays linear in the length of the text. A sum can differ by
    a token or two from the count of the joined text, so every finished text is counted once
    more, and the rare one above the target is cut again.
    """
    overhead = tok.count("")
    sep_tokens = max(tok.count(sep) - overhead, 0) if sep else 0
    cur: list[str] = []
    cur_tokens = overhead

    def flush() -> Iterator[str]:
        text = sep.join(cur).strip()
        if not text:
            return
        if tok.count(text) <= target:
            yield text
        else:
            yield from _cut(_WORD.findall(text), tok, target, window=target)

    for unit in units:
        n = tok.count(unit) - overhead
        if n + overhead > target:
            yield from flush()
            cur, cur_tokens = [], overhead
            yield from _split_long(unit, tok, target)
            continue
        added = n + (sep_tokens if cur else 0)
        if cur and cur_tokens + added > target:
            yield from flush()
            cur, cur_tokens = [], overhead
            added = n
        cur.append(unit)
        cur_tokens += added
    yield from flush()


def _sentences(text: str) -> list[str]:
    """`text` cut after each sentence end or line break; "".join() of the result is `text`."""
    pieces: list[str] = []
    start = 0
    for m in _BREAK.finditer(text):
        if m.end() > start:
            pieces.append(text[start : m.end()])
            start = m.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _split_long(text: str, tok: Tokenizer, target: int) -> Iterator[str]:
    """A paragraph above the target: by sentences first, then by words or characters."""
    sentences = _sentences(text)
    if len(sentences) > 1:
        yield from _pack(sentences, "", tok, target)
        return
    words = _WORD.findall(text)
    if len(words) > 1:
        yield from _cut(words, tok, target, window=target)
    else:
        yield from _cut(list(text), tok, target, window=8 * target)


def _cut(units: list[str], tok: Tokenizer, target: int, window: int) -> Iterator[str]:
    """Consecutive runs of `units`, each the longest that fits in `target` tokens.

    A binary search on the run length needs about log2(window) counts per run, each over at
    most `window` units. A word is at least one token, so a window of `target` words never
    cuts a run short. For characters a window of eight times the target covers text of up to
    eight characters per token; denser text only yields shorter runs, never longer ones.
    """
    i = 0
    while i < len(units):
        lo, hi = 0, min(len(units) - i, window)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tok.count("".join(units[i : i + mid])) <= target:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:  # a single word above the target, such as a long URL: cut by characters
            yield from _cut(list(units[i]), tok, target, window=8 * target)
            i += 1
            continue
        piece = "".join(units[i : i + lo]).strip()
        if piece:
            yield piece
        i += lo


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
    if target_tokens > max_tokens:
        raise ValueError(f"chunk target {target_tokens} exceeds the model limit {max_tokens}")
    chunks: list[Chunk] = []
    for section in _sections(body):
        for text in _pack(section.paragraphs, "\n\n", tok, target_tokens):
            chunks.append(Chunk(len(chunks), text, tok.count(text), section.heading_path))
    return chunks
