"""Stages 3 to 7 as one pipeline: PLAN what changed, then PARSE, CHUNK, EMBED and STORE it.

The unit of work is a group of documents whose chunks fill a few embedding batches. Three
threads overlap the stages, so the embeddings server never waits for parsing or for the
database: a producer parses and chunks groups, the caller's thread embeds them, a writer stores
them. Each group is written in one transaction, so stopping at any moment loses at most the
groups in flight and a rerun picks up from the stored hashes (premise P4).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .chunk import Chunk, chunk_document, embedding_input
from .config import Config
from .discover import Corpus, Tree, iter_files
from .embed import TEIClient
from .parse import Document, content_sha, parse_document
from .store import Store
from .tokenizer import Tokenizer

log = logging.getLogger(__name__)

Group = list[tuple[Document, list[Chunk]]]
_END = None


@dataclass(slots=True)
class CorpusPlan:
    corpus: Corpus
    corpus_id: int | None
    to_index: list[str]  # relative paths, new or changed
    to_delete: list[str]
    unchanged: int

    @property
    def total_files(self) -> int:
        return len(self.to_index) + self.unchanged


@dataclass(slots=True)
class Progress:
    """What the CLI or TUI shows. Fields are cumulative for the run."""

    corpus: str = ""
    corpus_done: int = 0
    corpus_total: int = 0
    files_done: int = 0
    files_total: int = 0
    chunks_done: int = 0
    started: float = field(default_factory=time.monotonic)
    message: str = ""

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def chunks_per_second(self) -> float:
        return self.chunks_done / self.elapsed if self.elapsed > 0 else 0.0


ProgressCallback = Callable[[Progress], None]


def plan_corpus(
    tree: Tree, corpus: Corpus, store: Store | None, max_files: int = 0
) -> CorpusPlan:
    """Diff the corpus on disk against what the store holds for it."""
    files = list(iter_files(tree.root, corpus))
    if max_files and len(files) > max_files:
        files = files[:max_files]
    on_disk = {f.as_posix(): f for f in files}
    stored: dict[str, str] = {}
    corpus_id: int | None = None
    if store is not None:
        row = store.corpus_rows().get(corpus.path)
        if row is not None:
            corpus_id = row.id
            stored = store.document_hashes(corpus_id)
    to_index: list[str] = []
    unchanged = 0
    for rel, p in on_disk.items():
        old = stored.get(rel)
        if old is None:
            to_index.append(rel)
            continue
        if old == content_sha((tree.root / p).read_bytes()):
            unchanged += 1
        else:
            to_index.append(rel)
    to_delete = sorted(set(stored) - set(on_disk))
    return CorpusPlan(corpus, corpus_id, to_index, to_delete, unchanged)


class Indexer:
    def __init__(
        self,
        cfg: Config,
        tree: Tree,
        store: Store,
        client: TEIClient,
        tokenizer: Tokenizer,
        progress: ProgressCallback | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.tree = tree
        self.store = store
        self.client = client
        self.tok = tokenizer
        self.progress_cb = progress or (lambda p: None)
        self.stop = stop or (lambda: False)
        self.progress = Progress()

    def _group_size(self) -> int:
        return max(self.cfg.batch_size * self.cfg.workers * 2, 16)

    def run(self, plans: list[CorpusPlan]) -> Progress:
        p = self.progress
        p.corpus_total = len(plans)
        p.files_total = sum(len(pl.to_index) for pl in plans)
        for pl in plans:
            if self.stop():
                break
            p.corpus = pl.corpus.path
            self._run_corpus(pl)
            p.corpus_done += 1
            self.progress_cb(p)
        if p.chunks_done and p.elapsed > 30:
            self.store.set_meta(chunks_per_second=f"{p.chunks_per_second:.2f}")
        return p

    # --- one corpus, three threads ---

    def _run_corpus(self, pl: CorpusPlan) -> None:
        cid = self.store.upsert_corpus(pl.corpus, pl.total_files)
        if pl.to_delete:
            n = self.store.delete_documents(cid, pl.to_delete)
            log.info("%s: removed %d deleted documents", pl.corpus.path, n)
        if not pl.to_index:
            if not self.stop():
                self.store.complete_corpus(cid)
            return

        parsed: queue.Queue = queue.Queue(maxsize=2)
        embedded: queue.Queue = queue.Queue(maxsize=2)
        errors: list[BaseException] = []
        abort = threading.Event()

        def stopping() -> bool:
            return abort.is_set() or self.stop()

        producer = threading.Thread(
            target=self._produce, args=(pl, parsed, errors, abort, stopping), daemon=True)
        writer = threading.Thread(
            target=self._write, args=(cid, embedded, errors, abort), daemon=True)
        producer.start()
        writer.start()
        try:
            while True:
                group = parsed.get()
                if group is _END:
                    break
                if stopping():
                    continue  # drain the producer without embedding
                try:
                    vectors = self._embed(group)
                except BaseException as exc:  # noqa: BLE001 - reported below
                    errors.append(exc)
                    abort.set()
                    continue
                embedded.put((group, vectors))
        finally:
            embedded.put(_END)
            producer.join()
            writer.join()
        if errors:
            raise errors[0]
        if not self.stop():
            self.store.complete_corpus(cid)

    def _produce(self, pl: CorpusPlan, out: queue.Queue, errors: list, abort: threading.Event,
                 stopping: Callable[[], bool]) -> None:
        group: Group = []
        pending = 0
        try:
            for rel in pl.to_index:
                if stopping():
                    break
                doc = parse_document(rel, pl.corpus.path, (self.tree.root / rel).read_bytes())
                chunks = chunk_document(doc, self.tok, self.cfg.chunk_tokens, self.cfg.max_tokens)
                group.append((doc, chunks))
                pending += len(chunks)
                if pending >= self._group_size():
                    out.put(group)
                    group, pending = [], 0
            if group and not stopping():
                out.put(group)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            abort.set()
        finally:
            out.put(_END)

    def _embed(self, group: Group) -> list[list[float]]:
        prefix = self.cfg.document_prefix
        texts = [embedding_input(doc, ch, prefix) for doc, chunks in group for ch in chunks]
        return self.client.embed_many(texts, self.cfg.batch_size, self.cfg.workers)

    def _write(self, corpus_id: int, inp: queue.Queue, errors: list,
               abort: threading.Event) -> None:
        p = self.progress
        try:
            while True:
                item = inp.get()
                if item is _END:
                    break
                if abort.is_set():
                    continue
                group, vectors = item
                items = []
                i = 0
                for doc, chunks in group:
                    items.append((doc, chunks, vectors[i : i + len(chunks)]))
                    i += len(chunks)
                self.store.write_documents(corpus_id, items)
                p.files_done += len(group)
                p.chunks_done += len(vectors)
                self.progress_cb(p)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            abort.set()
            # keep draining so the embedding thread never blocks on a full queue
            while inp.get() is not _END:
                pass


def dry_run_counts(tree: Tree, corpus: Corpus, tok: Tokenizer, cfg: Config,
                   max_files: int = 0) -> tuple[int, int, int]:
    """(files, chunks, tokens) a corpus would produce: parses and chunks, embeds nothing."""
    files = chunks = tokens = 0
    for rel in iter_files(tree.root, corpus):
        if max_files and files >= max_files:
            break
        doc = parse_document(rel.as_posix(), corpus.path, (tree.root / rel).read_bytes())
        cs = chunk_document(doc, tok, cfg.chunk_tokens, cfg.max_tokens)
        files += 1
        chunks += len(cs)
        tokens += sum(c.token_count for c in cs)
    return files, chunks, tokens


def resolve_root(cfg: Config) -> Path:
    root = cfg.global_law
    if not (root / "manifest.json").is_file() and not (root / ".gitmodules").is_file():
        raise FileNotFoundError(
            f"no global-law tree at {root}; set GLOBAL_LAW_PATH or run from next to it"
        )
    return root
