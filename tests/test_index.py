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
    """Just enough of Store for plan_corpus: one corpus holding the given files."""

    def __init__(self, hashes: dict[str, str], commit=None, completed=False) -> None:
        from global_law_vector.store import CorpusState

        self.hashes = hashes
        self.state = CorpusState(1, commit, completed, len(hashes))

    def corpus_state(self, path):
        return self.state

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


def _git(cwd, *args):
    import subprocess

    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _corpus_repo(tmp_path, files: dict[str, str]):
    """A corpus checkout that is its own git repository, as a global-law submodule is."""
    from global_law_vector.discover import Corpus, Tree

    corpus = Corpus.from_path("malta/national/acts")
    root = tmp_path / corpus.path
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    for name, text in files.items():
        (root / name).write_text(text)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "v1")
    return Tree(root=tmp_path, corpora=[corpus]), corpus, root


def _head(root) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_corpus_complete_at_the_pinned_commit_is_not_read(tmp_path):
    tree, corpus, root = _corpus_repo(tmp_path, {"a.md": "# A\n\nOne.\n"})
    corpus.commit_sha = _head(root)
    (root / "a.md").unlink()  # would be planned as a deletion if the corpus were read
    plan = plan_corpus(tree, corpus, PlannedStore({"x": "y"}, corpus.commit_sha, completed=True))
    assert plan.up_to_date and plan.to_index == [] and plan.to_delete == []


def test_new_commit_reads_only_the_files_git_reports_changed(tmp_path):
    tree, corpus, root = _corpus_repo(
        tmp_path, {"a.md": "# A\n\nOne.\n", "b.md": "# B\n\nTwo.\n", "c.md": "# C\n"}
    )
    old = _head(root)
    (root / "b.md").write_text("# B\n\nTwo, amended.\n")
    (root / "c.md").unlink()
    (root / "d.md").write_text("# D\n\nNew.\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "v2")
    corpus.commit_sha = _head(root)
    rel = {n: f"{corpus.path}/{n}" for n in ("a.md", "b.md", "c.md", "d.md")}
    # Every stored hash is wrong: a file that is hashed shows up as changed. Only b.md may be.
    stored = {rel["a.md"]: "stale", rel["b.md"]: "stale", rel["c.md"]: "stale"}
    plan = plan_corpus(tree, corpus, PlannedStore(stored, old, completed=True))
    assert sorted(plan.to_index) == [rel["b.md"], rel["d.md"]]
    assert plan.to_delete == [rel["c.md"]] and plan.unchanged == 1


def test_interrupted_run_rehashes_everything(tmp_path):
    tree, corpus, root = _corpus_repo(tmp_path, {"a.md": "# A\n\nOne.\n"})
    corpus.commit_sha = _head(root)
    stored = {f"{corpus.path}/a.md": "stale"}
    plan = plan_corpus(tree, corpus, PlannedStore(stored, corpus.commit_sha, completed=False))
    assert plan.to_index == [f"{corpus.path}/a.md"] and not plan.up_to_date


def test_limited_run_does_not_mark_the_corpus_complete(tree, server):
    store = FakeStore()
    full = _plans(tree)[0]
    limited = CorpusPlan(full.corpus, None, full.to_index, [], 0, limited=True)
    Indexer(_cfg(tree), tree, store, TEIClient(server), HeuristicTokenizer()).run([limited])
    assert store.writes >= 1 and store.completed == []
