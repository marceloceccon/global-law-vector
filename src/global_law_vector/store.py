"""Stage 7, STORE: PostgreSQL + pgvector.

One schema (default `global_law`) with corpus, document, chunk and index_meta tables, as in the
specification. Chunks go in through COPY. The index records the model, dimensions, vector type
and chunker version; opening it with other settings raises (premise P5).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib import resources

import psycopg
from psycopg import sql

from .chunk import CHUNKER_VERSION, Chunk
from .discover import Corpus
from .embed import vector_literal
from .parse import Document

log = logging.getLogger(__name__)

OPS = {"vector": "vector_cosine_ops", "halfvec": "halfvec_cosine_ops"}


class IndexMismatch(RuntimeError):
    pass


@dataclass(slots=True)
class CorpusRow:
    id: int
    path: str
    commit_sha: str | None
    file_count: int | None
    indexed_at: datetime
    completed_at: datetime | None
    documents: int
    chunks: int


@dataclass(slots=True)
class Hit:
    score: float
    corpus_path: str
    country: str
    path: str
    title: str | None
    citation: str | None
    heading_path: str | None
    text: str
    commit_sha: str | None
    ordinal: int


class Store:
    def __init__(self, dsn: str, schema: str = "global_law", vector_type: str = "halfvec",
                 dims: int = 768) -> None:
        if vector_type not in OPS:
            raise ValueError(f"vector_type must be one of {sorted(OPS)}")
        self.dsn = dsn
        self.schema = schema
        self.vector_type = vector_type
        self.dims = dims
        self.conn: psycopg.Connection = psycopg.connect(dsn, autocommit=False)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _t(self, table: str) -> sql.Composed:
        return sql.Identifier(self.schema, table)

    # --- schema and metadata ---

    def migrate(self) -> None:
        text = resources.files("global_law_vector").joinpath("sql/001_schema.sql").read_text()
        text = text.replace("{schema}", sql.Identifier(self.schema).as_string(self.conn))
        text = text.replace("{vector_type}", self.vector_type).replace("{dims}", str(self.dims))
        with self.conn.cursor() as cur:
            cur.execute(text)
        self.conn.commit()

    def get_meta(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT key, value FROM {}").format(self._t("index_meta")))
            return dict(cur.fetchall())

    def set_meta(self, **values: str | int | float) -> None:
        with self.conn.cursor() as cur:
            for k, v in values.items():
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (key, value) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                    ).format(self._t("index_meta")),
                    (k, str(v)),
                )
        self.conn.commit()

    def ensure_meta(self, model: str, tokenizer: str, chunk_tokens: int) -> None:
        """Record the index settings on first use; refuse to mix settings afterwards."""
        expected = {
            "model": model,
            "dims": str(self.dims),
            "vector_type": self.vector_type,
            "chunker_version": CHUNKER_VERSION,
        }
        current = self.get_meta()
        if not current:
            self.set_meta(**expected, tokenizer=tokenizer, chunk_tokens=chunk_tokens,
                          created_at=datetime.now().isoformat(timespec="seconds"))
            return
        for k, v in expected.items():
            if current.get(k) != v:
                raise IndexMismatch(
                    f"index was built with {k}={current.get(k)!r}, configured {v!r}; "
                    "use another schema (GLV_SCHEMA) or reset the index"
                )

    # --- corpora ---

    def upsert_corpus(self, corpus: Corpus, file_count: int | None) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} (path, country, code, level, name, language, commit_sha,"
                    " record_to, file_count, indexed_at, completed_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now(), NULL)"
                    " ON CONFLICT (path) DO UPDATE SET code = EXCLUDED.code, name = EXCLUDED.name,"
                    " language = EXCLUDED.language, commit_sha = EXCLUDED.commit_sha,"
                    " record_to = EXCLUDED.record_to, file_count = EXCLUDED.file_count,"
                    " indexed_at = now(), completed_at = NULL RETURNING id"
                ).format(self._t("corpus")),
                (corpus.path, corpus.country, corpus.code, corpus.level,
                 corpus.label or corpus.name, corpus.language, corpus.commit_sha,
                 corpus.record_to, file_count),
            )
            cid = cur.fetchone()[0]
        self.conn.commit()
        return cid

    def complete_corpus(self, corpus_id: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("UPDATE {} SET completed_at = now(), indexed_at = now() WHERE id = %s")
                .format(self._t("corpus")), (corpus_id,))
        self.conn.commit()

    def corpus_rows(self) -> dict[str, CorpusRow]:
        q = sql.SQL(
            "SELECT c.id, c.path, c.commit_sha, c.file_count, c.indexed_at, c.completed_at,"
            " (SELECT count(*) FROM {doc} d WHERE d.corpus_id = c.id),"
            " (SELECT count(*) FROM {chunk} k JOIN {doc} d ON d.id = k.document_id"
            "   WHERE d.corpus_id = c.id)"
            " FROM {corpus} c"
        ).format(doc=self._t("document"), chunk=self._t("chunk"), corpus=self._t("corpus"))
        with self.conn.cursor() as cur:
            cur.execute(q)
            return {r[1]: CorpusRow(*r) for r in cur.fetchall()}

    # --- documents and chunks ---

    def document_hashes(self, corpus_id: int) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT path, content_sha FROM {} WHERE corpus_id = %s")
                        .format(self._t("document")), (corpus_id,))
            return dict(cur.fetchall())

    def delete_documents(self, corpus_id: int, paths: Iterable[str]) -> int:
        paths = list(paths)
        if not paths:
            return 0
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {} WHERE corpus_id = %s AND path = ANY(%s)")
                        .format(self._t("document")), (corpus_id, paths))
            n = cur.rowcount
        self.conn.commit()
        return n

    def write_documents(
        self,
        corpus_id: int,
        items: Sequence[tuple[Document, list[Chunk], list[list[float]]]],
    ) -> None:
        """Replace each document and its chunks in one transaction."""
        if not items:
            return
        doc_t, chunk_t = self._t("document"), self._t("chunk")
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {} WHERE corpus_id = %s AND path = ANY(%s)")
                        .format(doc_t), (corpus_id, [d.path for d, _, _ in items]))
            ids: list[int] = []
            for doc, _, _ in items:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (corpus_id, path, title, citation, front_matter,"
                        " content_sha, byte_size) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id"
                    ).format(doc_t),
                    (corpus_id, doc.path, doc.title, doc.citation,
                     json.dumps(doc.front_matter, ensure_ascii=False) if doc.front_matter else None,
                     doc.content_sha, doc.byte_size),
                )
                ids.append(cur.fetchone()[0])
            copy_q = sql.SQL(
                "COPY {} (document_id, ordinal, heading_path, text, token_count, embedding)"
                " FROM STDIN"
            ).format(chunk_t)
            with cur.copy(copy_q) as copy:
                for doc_id, (_, chunks, vectors) in zip(ids, items, strict=True):
                    for ch, vec in zip(chunks, vectors, strict=True):
                        copy.write_row((doc_id, ch.ordinal, ch.heading_path, ch.text,
                                        ch.token_count, vector_literal(vec)))
        self.conn.commit()

    # --- index and search ---

    def build_hnsw(self, m: int = 16, ef_construction: int = 64,
                   maintenance_work_mem: str = "2GB", parallel: bool = True) -> None:
        """Build the HNSW index once, after the bulk load.

        A parallel build keeps its graph in dynamic shared memory sized by
        maintenance_work_mem. Postgres in Docker has a 64 MB /dev/shm unless the container
        sets shm_size, and then the build fails with "could not resize shared memory
        segment … No space left on device". In that case it is retried without parallel
        workers, which uses ordinary memory and is a few times slower.
        """
        create = sql.SQL(
            "CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw ON {} USING hnsw"
            " (embedding {}) WITH (m = {}, ef_construction = {})"
        ).format(self._t("chunk"), sql.SQL(OPS[self.vector_type]),
                 sql.Literal(m), sql.Literal(ef_construction))
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql.SQL("SET maintenance_work_mem = {}")
                            .format(sql.Literal(maintenance_work_mem)))
                if not parallel:
                    cur.execute("SET max_parallel_maintenance_workers = 0")
                cur.execute(create)
            self.conn.commit()
        except psycopg.Error as exc:
            self.conn.rollback()
            if parallel and "shared memory" in str(exc):
                log.warning("parallel HNSW build failed (%s); retrying without parallel"
                            " workers. Give the Postgres container shm_size: 2gb to avoid this.",
                            str(exc).strip().splitlines()[0])
                return self.build_hnsw(m, ef_construction, maintenance_work_mem, parallel=False)
            raise

    def has_hnsw(self) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
                        (self.schema, "chunk_embedding_hnsw"))
            return cur.fetchone() is not None

    def search(self, embedding: Sequence[float], top: int = 10,
               countries: Sequence[str] = (), corpora: Sequence[str] = ()) -> list[Hit]:
        where = [sql.SQL("TRUE")]
        params: list[object] = []
        if countries:
            where.append(sql.SQL("c.country = ANY(%s)"))
            params.append(list(countries))
        if corpora:
            where.append(sql.SQL("c.path = ANY(%s)"))
            params.append(list(corpora))
        cast = sql.SQL(self.vector_type)
        q = sql.SQL(
            "SELECT 1 - (k.embedding <=> %s::{cast}) AS score, c.path, c.country, d.path,"
            " d.title, d.citation, k.heading_path, k.text, c.commit_sha, k.ordinal"
            " FROM {chunk} k JOIN {doc} d ON d.id = k.document_id"
            " JOIN {corpus} c ON c.id = d.corpus_id"
            " WHERE {where} ORDER BY k.embedding <=> %s::{cast} LIMIT %s"
        ).format(cast=cast, chunk=self._t("chunk"), doc=self._t("document"),
                 corpus=self._t("corpus"), where=sql.SQL(" AND ").join(where))
        lit = vector_literal(embedding)
        with self.conn.cursor() as cur:
            cur.execute(q, [lit, *params, lit, top])
            return [Hit(*r) for r in cur.fetchall()]

    def reset(self) -> None:
        """Drop the whole schema. Irreversible; the CLI asks first."""
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE")
                        .format(sql.Identifier(self.schema)))
        self.conn.commit()
