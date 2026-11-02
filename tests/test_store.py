"""Store against a real PostgreSQL with pgvector, in a throwaway schema.

Needs no embeddings server: vectors are written by hand in three dimensions. Runs when the
database in DATABASE_URL is reachable and skips otherwise; with GLV_REQUIRE_DB=1 (set in CI)
an unreachable database fails the run instead of skipping it.
"""

from __future__ import annotations

import os
import uuid

import pytest

from global_law_vector.chunk import Chunk
from global_law_vector.config import Config
from global_law_vector.discover import Corpus
from global_law_vector.parse import Document
from global_law_vector.store import IndexMismatch, Store

DIMS = 3


@pytest.fixture
def store():
    psycopg = pytest.importorskip("psycopg")
    dsn = Config.from_env().database_url
    try:
        psycopg.connect(dsn, connect_timeout=3).close()
    except psycopg.Error as exc:
        if os.environ.get("GLV_REQUIRE_DB") == "1":
            pytest.fail(f"GLV_REQUIRE_DB=1 but the database is unreachable: {exc}")
        pytest.skip("no database reachable at DATABASE_URL")
    schema = f"glv_test_{uuid.uuid4().hex[:8]}"
    s = Store(dsn, schema, "halfvec", DIMS)
    try:
        yield s
    finally:
        s.conn.rollback()
        s.reset()
        s.close()


def _corpus(path: str, commit: str = "c1") -> Corpus:
    c = Corpus.from_path(path)
    c.commit_sha = commit
    return c


def _items(corpus: Corpus, vectors: dict[str, list[float]]):
    """One single-chunk document per name, embedded as given."""
    out = []
    for name, vec in vectors.items():
        doc = Document(
            path=f"{corpus.path}/{name}.md",
            corpus_path=corpus.path,
            title=name,
            body=f"text of {name}",
            content_sha=f"sha-{name}",
            byte_size=10,
        )
        out.append((doc, [Chunk(0, doc.body, 3)], [vec]))
    return out


def _load(store: Store) -> tuple[int, int]:
    store.migrate()
    store.ensure_meta("m", "heuristic", 1024, token_counting="heuristic")
    malta, france = _corpus("malta/national/acts"), _corpus("france/national/codes")
    mid = store.upsert_corpus(malta, 3)
    store.write_documents(
        mid, _items(malta, {"near": [1, 0, 0], "mid": [0.7, 0.7, 0], "far": [0, 0, 1]})
    )
    store.complete_corpus(mid)
    fid = store.upsert_corpus(france, 2)
    store.write_documents(fid, _items(france, {"closest": [1, 0.01, 0], "other": [0, 1, 0]}))
    store.complete_corpus(fid)
    return mid, fid


def test_migrate_is_idempotent_and_detectable(store):
    assert not store.schema_exists()
    store.migrate()
    store.migrate()
    assert store.schema_exists()


def test_settings_are_recorded_and_enforced(store):
    store.migrate()
    store.ensure_meta("m", "tok", 1024, "", "exact")
    store.ensure_meta("m", "another-name", 1024, "", "exact")  # the name is informational
    with pytest.raises(IndexMismatch, match="chunk_tokens"):
        store.ensure_meta("m", "tok", 512, "", "exact")
    with pytest.raises(IndexMismatch, match="token_counting"):
        store.ensure_meta("m", "tok", 1024, "", "heuristic")
    with pytest.raises(IndexMismatch, match="document_prefix"):
        store.ensure_meta("m", "tok", 1024, "passage: ", "exact")


def test_completion_records_state_and_chunk_count(store):
    mid, _ = _load(store)
    state = store.corpus_state("malta/national/acts")
    assert state.id == mid and state.completed and state.commit_sha == "c1"
    assert store.corpus_state("nowhere/x/y") is None
    with store.conn.cursor() as cur:
        cur.execute(f"SELECT chunk_count FROM {store.schema}.corpus WHERE id = %s", (mid,))
        assert cur.fetchone()[0] == 3
    store.conn.rollback()
    store.upsert_corpus(_corpus("malta/national/acts", "c2"), 3)  # a new run starts
    state = store.corpus_state("malta/national/acts")
    assert not state.completed and state.commit_sha == "c2"


@pytest.mark.parametrize("with_hnsw", [False, True])
def test_search_orders_by_cosine_with_and_without_filters(store, with_hnsw):
    _load(store)
    if with_hnsw:
        store.build_hnsw()
        assert store.has_hnsw()
    everywhere = store.search([1, 0, 0], top=3)
    assert [h.title for h in everywhere] == ["near", "closest", "mid"]
    assert everywhere[0].score == pytest.approx(1.0, abs=1e-3)
    malta = store.search([1, 0, 0], top=10, countries=["malta"])
    assert [h.title for h in malta] == ["near", "mid", "far"]  # every Maltese chunk, in order
    assert {h.country for h in malta} == {"malta"}
    by_corpus = store.search([0, 1, 0], top=1, corpora=["france/national/codes"])
    assert [h.title for h in by_corpus] == ["other"]
    assert store.search([1, 0, 0], countries=["iceland"]) == []


def test_filter_size_falls_back_to_counting_without_chunk_counts(store):
    mid, _ = _load(store)
    with store.conn.cursor() as cur:
        cur.execute(f"UPDATE {store.schema}.corpus SET chunk_count = NULL WHERE id = %s", (mid,))
    store.conn.commit()
    assert [h.title for h in store.search([1, 0, 0], countries=["malta"])] == [
        "near",
        "mid",
        "far",
    ]


def test_failed_write_rolls_back_and_leaves_the_connection_usable(store):
    mid, _ = _load(store)
    malta = _corpus("malta/national/acts")
    bad = _items(malta, {"near": [0, 0, 1]})
    bad[0] = (bad[0][0], bad[0][1], [])  # a chunk without a vector
    with pytest.raises(ValueError):
        store.write_documents(mid, bad)
    assert store.document_hashes(mid)[f"{malta.path}/near.md"] == "sha-near"  # untouched
    assert store.search([1, 0, 0], top=1)[0].title == "near"


def test_delete_documents_removes_their_chunks(store):
    mid, _ = _load(store)
    assert store.delete_documents(mid, ["malta/national/acts/near.md"]) == 1
    assert "near" not in [h.title for h in store.search([1, 0, 0], top=10)]


def test_sampling_is_reproducible_and_capped_per_country(store):
    _load(store)
    a = store.sample_by_country(2, seed=7)
    assert a == store.sample_by_country(2, seed=7)
    per_country = {}
    for _, country, _, total in a:
        per_country[country] = per_country.get(country, 0) + 1
        assert total == {"malta": 3, "france": 2}[country]
    assert per_country == {"malta": 2, "france": 2}
    ids = [row[0] for row in a]
    fetched = [row for batch in store.iter_embeddings(ids, batch=1) for row in batch]
    assert sorted(i for i, _ in fetched) == sorted(ids)
    assert all(text.startswith("[") for _, text in fetched)
