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
    Indexer(_cfg(tree), tree, store, TEIClient("http://127.0.0.1:9", retries=0),
            HeuristicTokenizer()).run([pl])
    assert store.completed == [7] and store.writes == 0
