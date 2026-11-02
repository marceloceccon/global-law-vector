-- global-law-vector schema. Every statement is idempotent; Store.migrate() runs it on each `glv index`.
-- Store.migrate() fills the three placeholders in curly braces: schema, vector_type, dims.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.index_meta (
  key   text PRIMARY KEY,
  value text NOT NULL
);

CREATE TABLE IF NOT EXISTS {schema}.corpus (
  id            serial PRIMARY KEY,
  path          text UNIQUE NOT NULL,
  country       text NOT NULL,
  code          text,
  level         text NOT NULL,
  name          text,
  language      text,
  commit_sha    text,
  record_to     date,
  file_count    int,
  chunk_count   bigint,                 -- set when a run completes; read by filtered search
  indexed_at    timestamptz NOT NULL DEFAULT now(),
  completed_at  timestamptz
);
ALTER TABLE {schema}.corpus ADD COLUMN IF NOT EXISTS chunk_count bigint;

CREATE TABLE IF NOT EXISTS {schema}.document (
  id            bigserial PRIMARY KEY,
  corpus_id     int NOT NULL REFERENCES {schema}.corpus(id) ON DELETE CASCADE,
  path          text NOT NULL,
  title         text,
  citation      text,
  front_matter  jsonb,
  content_sha   text NOT NULL,
  byte_size     int NOT NULL,
  in_force_date date,                   -- reserved for indexing past versions; not filled yet
  UNIQUE (corpus_id, path)
);

CREATE TABLE IF NOT EXISTS {schema}.chunk (
  id            bigserial PRIMARY KEY,
  document_id   bigint NOT NULL REFERENCES {schema}.document(id) ON DELETE CASCADE,
  ordinal       int NOT NULL,
  heading_path  text,
  text          text NOT NULL,
  token_count   int NOT NULL,
  embedding     {vector_type}({dims}) NOT NULL,
  UNIQUE (document_id, ordinal)
);

-- The UNIQUE constraints above already index document(corpus_id, …) and chunk(document_id, …).
-- Earlier versions created separate indexes on those leading columns; they only cost writes.
DROP INDEX IF EXISTS {schema}.chunk_document_idx;
DROP INDEX IF EXISTS {schema}.document_corpus_idx;
