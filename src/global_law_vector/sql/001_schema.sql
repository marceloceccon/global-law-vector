-- global-law-vector schema, version 1.
-- Placeholders filled by store.py: {schema}, {vector_type}, {dims}.
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
  indexed_at    timestamptz NOT NULL DEFAULT now(),
  completed_at  timestamptz
);

CREATE TABLE IF NOT EXISTS {schema}.document (
  id            bigserial PRIMARY KEY,
  corpus_id     int NOT NULL REFERENCES {schema}.corpus(id) ON DELETE CASCADE,
  path          text NOT NULL,
  title         text,
  citation      text,
  front_matter  jsonb,
  content_sha   text NOT NULL,
  byte_size     int NOT NULL,
  in_force_date date,
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

CREATE INDEX IF NOT EXISTS chunk_document_idx ON {schema}.chunk (document_id);
CREATE INDEX IF NOT EXISTS document_corpus_idx ON {schema}.document (corpus_id);
