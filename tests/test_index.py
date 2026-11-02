"""The three-thread indexer against the stub embeddings server and an in-memory store."""

from __future__ import annotations

import threading
from http.server import HTTPServer

import pytest

from global_law_vector.config import Config
from global_law_vector.embed import EmbeddingError, TEIClient
from global_law_vector.index import CorpusPlan, Indexer, plan_corpus
from global_law_vector.tokenizer import HeuristicTokenizer

from .test_embed import _Stub


class FakeStore:
    def __init__(self) -> None:
        self.docs: dict[str, tuple] = {}
        self.completed: list[int] = []
        self.meta: dict = {}
        self.writes = 0
        self.fail_writes = False

    def upsert_corpus(self, corpus, file_count):
        return 7

    def delete_documents(self, cid, paths):
        return 0

    def write_documents(self, cid, items):
        if self.fail_writes:
            raise RuntimeError("disk full")
        self.writes += 1
        for doc, chunks, vectors in items:
            assert len(chunks) == len(vectors)
            self.docs[doc.path] = (doc, chunks, vectors)

    def complete_corpus(self, cid):
        self.completed.append(cid)

    def set_meta(self, **kw):
        self.meta.update(kw)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def _cfg(tree) -> Config:
    cfg = Config()
    cfg.global_law = tree.root
    cfg.batch_size = 2
    cfg.workers = 2  # group size = 8 chunks
    return cfg


def _plans(tree):
    return [plan_corpus(tree, c, None) for c in tree.corpora]


def test_indexes_every_file_through_the_pipeline(tree, server):
    _Stub.fail_next = 0
    store = FakeStore()
    ix = Indexer(_cfg(tree), tree, store, TEIClient(server), HeuristicTokenizer())
    seen = []
    ix.progress_cb = lambda p: seen.append((p.files_done, p.chunks_done))
    p = ix.run(_plans(tree))
    assert p.files_done == 7 and p.chunks_done == 7
    assert sorted(store.docs) == sorted(f for pl in _plans(tree) for f in pl.to_index)
    doc, chunks, vectors = store.docs["germany/bund/acts/solzg/0001.md"]
    assert len(vectors[0]) == 3 and vectors[0][0] > 0  # stub embeds length of the input
    assert len(store.completed) == 7
    assert seen[-1] == (7, 7)


def test_stop_flag_leaves_corpus_incomplete(tree, server):
    store = FakeStore()
    cfg = _cfg(tree)
    plans = _plans(tree)
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 3  # stop early in the first corpora

    ix = Indexer(cfg, tree, store, TEIClient(server), HeuristicTokenizer(), stop=stop)
    p = ix.run(plans)
    assert p.corpus_done < len(plans)
    assert len(store.completed) < len(plans)


def test_embedding_failure_propagates_and_writes_nothing_more(tree, server):
    store = FakeStore()
    cfg = _cfg(tree)
    client = TEIClient(server, retries=0)
    _Stub.fail_next = 50
    try:
        with pytest.raises(EmbeddingError):
            Indexer(cfg, tree, store, client, HeuristicTokenizer()).run(_plans(tree))
    finally:
        _Stub.fail_next = 0
    assert store.completed == []


def test_store_failure_propagates(tree, server):
    store = FakeStore()
    store.fail_writes = True
    with pytest.raises(RuntimeError, match="disk full"):
        Indexer(_cfg(tree), tree, store, TEIClient(server), HeuristicTokenizer()).run(_plans(tree))
    assert store.completed == []


def test_empty_plan_completes_without_embedding(tree, server):
    store = FakeStore()
    c = tree.corpora[0]
    pl = CorpusPlan(c, None, [], [], 3)
    Indexer(
        _cfg(tree), tree, store, TEIClient("http://127.0.0.1:9", retries=0), HeuristicTokenizer()
    ).run([pl])
    assert store.completed == [7] and store.writes == 0


class PlannedStore:
    """Just enough of Store for plan_corpus: one corpus already holding every file."""

    def __init__(self, hashes: dict[str, str]) -> None:
        self.hashes = hashes

    def corpus_id(self, path):
        return 1

    def document_hashes(self, corpus_id):
        return self.hashes


def test_max_files_limits_indexing_but_never_deletes(tmp_path):
    from global_law_vector.discover import Corpus, Tree

    corpus = Corpus.from_path("malta/national/acts")
    for i in range(3):
        f = tmp_path / corpus.path / f"art-{i}.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"# Article {i}\n\nText {i}.\n")
    tree = Tree(root=tmp_path, corpora=[corpus])
    full = plan_corpus(tree, corpus, None)
    assert len(full.to_index) == 3
    stored = {rel: "old-hash" for rel in full.to_index}
    plan = plan_corpus(tree, corpus, PlannedStore(stored), max_files=1)
    assert len(plan.to_index) == 1  # one changed file considered
    assert plan.to_delete == []  # the files beyond the limit still exist


def test_deleted_files_are_planned_for_deletion(tree):
    corpus = tree.corpora[0]
    plan = plan_corpus(tree, corpus, PlannedStore({"gone/file.md": "x"}))
    assert plan.to_delete == ["gone/file.md"]


def test_interrupt_while_embedding_stops_every_thread(tree, server):
    """Ctrl-C arrives in the main thread while the producer may be blocked on a full queue."""
    cfg = _cfg(tree)
    cfg.batch_size = cfg.workers = 1  # small groups, so the producer fills its queue
    ix = Indexer(cfg, tree, FakeStore(), TEIClient(server), HeuristicTokenizer())

    def interrupt(group):
        raise KeyboardInterrupt

    ix._embed = interrupt
    plan = [
        CorpusPlan(c, None, pl.to_index, [], 0)
        for c, pl in zip(tree.corpora, _plans(tree), strict=True)
    ]
    done = threading.Event()
    outcome: list[BaseException] = []

    def run() -> None:
        try:
            ix.run(plan)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            outcome.append(exc)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(10), "the indexer hung after an interrupt"
    assert isinstance(outcome[0], KeyboardInterrupt)
