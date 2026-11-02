"""End to end against the real services. Skipped unless both are reachable.

Set DATABASE_URL and TEI_URL (defaults: the containers from the README on localhost). The
test uses its own schema `glv_test` and drops it afterwards.
"""

from __future__ import annotations

import os

import pytest

from global_law_vector.config import Config
from global_law_vector.embed import TEIClient
from global_law_vector.index import Indexer, plan_corpus
from global_law_vector.search import search
from global_law_vector.tokenizer import HeuristicTokenizer

from .conftest import FIXTURE_TREE


def _services() -> Config | None:
    cfg = Config.from_env()
    cfg.global_law = FIXTURE_TREE
    cfg.schema = os.environ.get("GLV_TEST_SCHEMA", "glv_test")
    if not TEIClient(cfg.tei_url, retries=0, timeout=3, api_key=cfg.tei_api_key).health():
        return None
    try:
        import psycopg

        psycopg.connect(cfg.database_url, connect_timeout=3).close()
    except Exception:  # noqa: BLE001
        return None
    return cfg


@pytest.fixture(scope="module")
def cfg():
    c = _services()
    if c is None:
        pytest.skip("embeddings server or database not reachable")
    return c


def test_index_then_search(cfg, tree):
    from global_law_vector.embed import check_model
    from global_law_vector.store import Store

    client = TEIClient(cfg.tei_url, api_key=cfg.tei_api_key)
    info = client.info()
    cfg.model = info.model_id or cfg.model
    cfg.dims = info.dims or client.probe_dims()
    check_model(info, cfg.model, cfg.dims, client)
    store = Store(cfg.database_url, cfg.schema, cfg.vector_type, cfg.dims)
    try:
        store.reset()
        store.migrate()
        store.ensure_meta(cfg.model, "heuristic", cfg.chunk_tokens)
        plans = [plan_corpus(tree, c, store) for c in tree.corpora]
        assert sum(len(p.to_index) for p in plans) == 7
        p = Indexer(cfg, tree, store, client, HeuristicTokenizer()).run(plans)
        assert p.files_done == 7 and p.chunks_done == 7
        rows = store.corpus_rows()
        assert all(r.completed_at is not None for r in rows.values())
        # a rerun is a no-op
        plans = [plan_corpus(tree, c, store) for c in tree.corpora]
        assert sum(len(p.to_index) for p in plans) == 0
        assert sum(p.unchanged for p in plans) == 7
        store.build_hnsw(maintenance_work_mem="256MB")
        hits = search(
            store,
            client,
            "solidarity surcharge on income tax",
            top=3,
            query_prefix=cfg.query_prefix,
        )
        assert hits and hits[0].path == "germany/bund/acts/solzg/0001.md"
        hits = search(
            store,
            client,
            "prazo de pagamento do comissário",
            top=3,
            countries=["brazil"],
            query_prefix=cfg.query_prefix,
        )
        assert hits and all(h.country == "brazil" for h in hits)
        assert hits[0].path.endswith("codigo-civil/art-0700.md")
    finally:
        store.reset()
        store.close()
