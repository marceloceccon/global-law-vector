# global-law-vector — Specification

Version 0.1, 2026-10-31. Status: specification. No application code yet.

## 1. Core concept

`global-law` (https://github.com/lexifina/global-law) is the law of 211 jurisdictions as one git
history: one markdown file per provision, one commit per law taking effect. Git answers exactly
*what the law said on a date*. It cannot answer *which provisions, anywhere, are about a topic*.

`global-law-vector` is a Python utility that turns a chosen subset of that tree into a vectorised
index: it reads the provision files, embeds them through a local embeddings service, and stores
text, vectors and metadata in PostgreSQL with pgvector. The result is a semantic search layer
(RAG store) over world law that any application, or an LLM through a tool, can query.

Two things make it different from "embed a folder":

1. **Selection is a first-class feature.** The tree holds 4.5 million files in 107 corpora. Nobody
   indexes all of it on the first day. Choosing countries and corpora must be trivial: a selection
   file, CLI flags, or an interactive terminal UI that shows what each choice costs.
2. **The index knows where it came from.** Every chunk is traceable to a corpus, a commit, a file
   and a position, so a search hit can be read back from `global-law` exactly, on any date.

## 2. Premises

P1. **Read-only source.** The utility never writes inside `global-law`, never runs `git checkout`
    there, and never relies on the root repository's commit hashes (they change on every rebuild).
    It relies on `manifest.json`, `jurisdiction.json`, each corpus's pinned commit, and the files.

P2. **The law in force on the build date is version 1.** Each corpus is indexed at its pinned
    HEAD. Historical versions (the law on past dates) are designed for, not built: the schema
    carries the corpus commit and an in-force date so a later version can index each law's commit.

P3. **Selection before cost.** No embedding happens until the user has chosen countries or corpora
    and seen the file count, byte size and estimated time of that choice.

P4. **Idempotent and resumable.** A run can be stopped at any point and restarted. A second run
    over an unchanged selection writes nothing. Change detection is per corpus (commit SHA) and per
    file (content hash).

P5. **The model is configuration, recorded in the database.** Model id, dimensions, context length
    and chunker version are stored with the index. A change in any of them is a deliberate
    re-index, never silent corruption.

P6. **Multilingual by default.** The corpus is mostly not English (pt, de, cs, es, fr, mt, ka, …).
    The default model must be multilingual; English-only models are an explicit opt-in.

P7. **Open-source quality.** Typed Python 3.11+, small dependency set, unit tests on fixtures taken
    from real corpora, structured logs, a README that gets a stranger from clone to first search in
    five minutes, and a licence compatible with `global-law` (CC0).

P8. **Infrastructure is external and local.** The embeddings server (Hugging Face
    text-embeddings-inference) and pgvector run as Docker containers defined in a sibling project
    (`global-law-utility/docker-compose.yml`). This repository talks to them over HTTP and
    PostgreSQL and does not own them, so the same containers can serve other projects.

## 3. Information flow

```
                 global-law (read-only, git tree)
   ┌──────────────────────────────────────────────────────────────┐
   │ manifest.json   jurisdiction.json   <country>/<level>/<corpus>/**/*.md │
   └───────────┬──────────────────────────────────┬───────────────┘
               │                                  │
               ▼                                  │
   ┌───────────────────────┐                      │
   │ 1. DISCOVER           │                      │
   │ corpora, pinned SHA,  │                      │
   │ file lists, counts,   │                      │
   │ bytes, language       │                      │
   └───────────┬───────────┘                      │
               ▼                                  │
   ┌───────────────────────┐    selection.toml    │
   │ 2. SELECT             │◄── CLI flags         │
   │ countries / corpora   │◄── TUI checklist     │
   │ shows cost estimate   │                      │
   └───────────┬───────────┘                      │
               ▼                                  ▼
   ┌───────────────────────┐    ┌──────────────────────────────┐
   │ 3. PLAN               │    │ 4. PARSE                      │
   │ diff selection vs DB: │───►│ front matter, title, body,    │
   │ new / changed /       │    │ citation; skip non-law files  │
   │ deleted files         │    └──────────────┬───────────────┘
   └───────────────────────┘                   ▼
                                ┌──────────────────────────────┐
                                │ 5. CHUNK                      │
                                │ token-aware; headings →       │
                                │ paragraphs; ordinal + heading │
                                │ path kept per chunk           │
                                └──────────────┬───────────────┘
                                               ▼
   ┌───────────────────────┐    ┌──────────────────────────────┐
   │ TEI embeddings server │◄───│ 6. EMBED                      │
   │ (Docker, HTTP /embed) │───►│ batches, N workers, retries,  │
   └───────────────────────┘    │ content-hash cache            │
                                └──────────────┬───────────────┘
                                               ▼
   ┌───────────────────────┐    ┌──────────────────────────────┐
   │ pgvector (Docker)     │◄───│ 7. STORE                      │
   │ corpus / document /   │    │ COPY to staging, upsert,      │
   │ chunk / index_meta    │    │ delete removed, HNSW build    │
   └───────────┬───────────┘    └──────────────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │ 8. SEARCH             │  query → embed → cosine top-k →
   │ CLI / TUI / library   │  path, heading, score, citation,
   │                       │  corpus commit (read back in git)
   └───────────────────────┘
```

Stages 1 and 2 cost seconds. Stage 6 is the bottleneck and the only one that scales with the
selection in hours.

## 4. The data source, as measured on 2026-10-29's build

| Fact | Value |
| --- | --- |
| Jurisdictions in `jurisdiction.json` | 211 |
| Corpora pinned as submodules (`manifest.json`) | 107 |
| Markdown files (excluding `.git`) | 4,545,302 |
| Total markdown bytes | ~5.5 GB (tree is 21 GB with git objects) |
| Mean file size | ~1.2 KB, roughly 300 tokens |
| Files over 200 KB | 585 |

File shapes differ per corpus, and the parser must tolerate all of them:

- **Heading + body** (Germany `germany/bund/acts/solzg/0001.md`): `# § 1 …` then paragraphs.
- **YAML front matter + heading + whole act** (Czechia `czechia/national/acts/sb/1919/213.md`):
  `citace`, `nazev`, `druh`, `eli`, then `###` sections. These are the large files.
- **Bare text, no heading** (São Paulo `brazil/sao-paulo/acts/lei-1990-07004/art-0001.md`): one
  article per file; the act and article number live only in the path.
- **Heading + body + Source line** (California `united-states/california/statutes/ins/1182.md`).
- **Not law, skipped**: `README.md`, `REPRODUCE.md`, `SOURCES.md`, `CHANGELOG.md`, `LICENSE.md`,
  `MCP.md`, and everything under `judgments/`, `visualisations/`, `tools/`, `tests/`.

Levels below country are real: Brazil has `national` plus thirteen state corpora (São Paulo alone
is 90,847 files); the United States has federal and state corpora; Bosnia has entity and canton
corpora. Selection therefore works on **country → corpus**, not country alone.

## 5. Selection

Three ways to say what to index. They produce the same object, a list of corpus paths, and the
TUI and CLI both write it to the selection file so a choice made interactively is reproducible.

### 5.1 Selection file: `selection.toml`

```toml
# global-law-vector selection
[include]
countries = ["brazil", "germany", "malta"]          # every corpus under these countries
corpora   = ["united-states/federal/usc"]            # individual corpora by manifest path

[exclude]
corpora   = ["brazil/sao-paulo/acts"]                # subtract after include

[limits]
max_files_per_corpus = 0                             # 0 = no limit; >0 = sample for a dry run
```

### 5.2 CLI

```sh
glv select --country brazil --country germany --corpus united-states/federal/usc
glv select --exclude brazil/sao-paulo/acts
glv select --all                 # every pinned corpus
glv select --show                # the resolved list with files, MB, est. chunks, est. time
glv index                        # index the selection; --dry-run plans only
glv index --country malta        # one-off override without touching the file
glv status                       # per corpus: selected / indexed / stale / not indexed
glv search "prazo de prescrição" --country brazil --top 10
```

### 5.3 TUI (Textual)

`glv tui` opens a three-screen terminal application built on Textual:

```
┌ global-law-vector ──────────────────────────────── model: multilingual-e5-small ┐
│ Filter: bra_                                                                     │
│                                                                                  │
│ [x] brazil                       14 corpora   365,210 files   1.9 GB   ~1h 05m  │
│   [x] national/acts               12,114 files   61 MB   ● indexed 2026-10-30    │
│   [ ] sao-paulo/acts              90,847 files  444 MB   ○ not indexed            │
│   [x] minas-gerais/acts           31,002 files  151 MB   ◐ stale (new commit)     │
│   …                                                                              │
│ [ ] bosnia-and-herzegovina        8 corpora   …                                   │
│ [x] germany                       1 corpus    113,481 files  531 MB   ~0h 20m     │
│                                                                                  │
│ Selected: 9 corpora · 160,597 files · 743 MB · ~160k chunks · est. 0h 28m         │
├──────────────────────────────────────────────────────────────────────────────────┤
│ space toggle  a all  n none  / filter  s save selection  r run  q quit            │
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Select screen**: a tree of country → corpus with checkboxes, a type-to-filter box, and live
  totals (files, bytes, estimated chunks and time from a measured chunks-per-second figure).
  Each corpus shows its index state: not indexed, indexed with date, or stale (corpus commit
  differs from the stored one).
- **Run screen**: one progress bar per corpus plus a total, throughput, ETA, errors, and a log
  pane. Pause and resume; cancelling is safe (P4).
- **Search screen**: a query box, optional country filter, and a results list; selecting a result
  shows the chunk text and the `global-law` path and commit to read it back.

Textual is the only UI dependency; the TUI is optional and the package works without it.

## 6. Components

```
global-law-vector/
├── specification.md          this document
├── README.md                 five-minute path
├── pyproject.toml            package `global_law_vector`, console script `glv`
├── selection.toml            the user's selection (git-ignored example provided)
├── src/global_law_vector/
│   ├── config.py             env / .env: GLOBAL_LAW_PATH, DATABASE_URL, TEI_URL, model, dims
│   ├── discover.py           manifest → corpora with SHA, file lists, counts, bytes
│   ├── selection.py          selection.toml read/write, resolve to corpus paths, cost estimate
│   ├── parse.py              front matter, title, body, citation, skip rules
│   ├── chunk.py              token-aware splitting with heading paths
│   ├── embed.py              TEI client: batching, concurrency, retry, hash cache
│   ├── store.py              migrations, COPY loader, upsert, delete, HNSW build
│   ├── search.py             query embedding + cosine top-k with filters
│   ├── cli.py                `select`, `index`, `status`, `search`, `tui`
│   └── tui/                  Textual app: select, run, search screens
├── sql/                      versioned schema migrations
└── tests/
    ├── fixtures/             real files copied from six corpora of different shapes
    ├── test_parse.py, test_chunk.py, test_selection.py, test_discover.py
    └── test_integration.py   needs the Docker services; skipped otherwise
```

Dependencies: `psycopg[binary]`, `httpx`, `pyyaml`, `tomli` (3.10 compat) or stdlib `tomllib`,
a tokenizer (`tokenizers`), `typer` for the CLI, `textual` for the TUI. No ML frameworks.

## 7. Storage schema (pgvector)

```sql
CREATE SCHEMA IF NOT EXISTS global_law;

CREATE TABLE global_law.corpus (
  id            serial PRIMARY KEY,
  path          text UNIQUE NOT NULL,        -- "germany/bund/acts"
  country       text NOT NULL,               -- "germany"
  code          text,                        -- "DE"
  level         text NOT NULL,               -- "bund" / "national" / "sao-paulo"
  name          text,
  language      text,
  commit_sha    text NOT NULL,               -- pinned submodule commit at index time
  record_to     date,
  file_count    int,
  indexed_at    timestamptz NOT NULL
);

CREATE TABLE global_law.document (
  id            bigserial PRIMARY KEY,
  corpus_id     int NOT NULL REFERENCES global_law.corpus(id) ON DELETE CASCADE,
  path          text NOT NULL,               -- relative to the global-law root
  title         text,
  citation      text,                        -- from front matter when present ("213/1919 Sb.")
  front_matter  jsonb,
  content_sha   text NOT NULL,               -- sha256 of file bytes; skip when unchanged
  byte_size     int NOT NULL,
  in_force_date date,                        -- optional in v1 (needs git log per file)
  UNIQUE (corpus_id, path)
);

CREATE TABLE global_law.chunk (
  id            bigserial PRIMARY KEY,
  document_id   bigint NOT NULL REFERENCES global_law.document(id) ON DELETE CASCADE,
  ordinal       int NOT NULL,
  heading_path  text,                        -- "§ 1 > (2)"
  text          text NOT NULL,
  token_count   int NOT NULL,
  embedding     vector(384) NOT NULL,        -- dims from config; halfvec if space matters
  UNIQUE (document_id, ordinal)
);

CREATE TABLE global_law.index_meta (         -- model id, dims, max tokens, chunker version, built_at
  key   text PRIMARY KEY,
  value text NOT NULL
);

-- after the bulk load:
CREATE INDEX chunk_embedding_hnsw ON global_law.chunk USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunk_document_idx   ON global_law.chunk (document_id);
CREATE INDEX document_corpus_idx  ON global_law.document (corpus_id);
```

Sizing at 384 dimensions for the whole tree: about 5 million chunks × 1.5 KB ≈ 7.5 GB of vectors
plus a comparable HNSW index. `halfvec` halves both. A typical selection is far smaller.

## 8. Pipeline and performance

- Embedding on CPU through TEI runs on the order of 100–300 chunks/s for a small model, so the
  whole tree is 5–15 hours and Brazil alone about one hour. The cost estimate in the TUI and in
  `glv select --show` uses a chunks-per-second figure measured on the first run and stored in
  `index_meta`.
- Load first, build HNSW once, with `maintenance_work_mem` raised for the build. Re-indexing a
  changed corpus updates rows under the existing index.
- A rerun compares corpus SHA and file hashes and touches only new, changed or deleted files.

## 9. Open decisions

1. **Embedding model.** Default proposed: `intfloat/multilingual-e5-small` (384 dims, 512 tokens,
   same cost class as the English `snowflake-arctic-embed-xs` currently in the compose file).
   `BAAI/bge-m3` is better and has an 8k context but is several times slower on CPU.
2. **Starting selection.** Proposed: Brazil national plus one or two states, Germany, Malta and
   the US federal code, to validate quality across languages before any multi-hour run.
3. **Vector type.** `vector(384)` or `halfvec(384)`. Proposed: `halfvec` from the start.
4. **In-force date per document in v1.** Costs one `git log` per file. Proposed: off by default,
   `--with-dates` flag.

## 10. Milestones

| # | Deliverable | Done when |
| --- | --- | --- |
| M0 | Specification agreed, model and starting selection chosen | Open decisions closed |
| M1 | `discover`, `selection`, `parse`, `chunk` with fixture tests; `glv select --show` and `glv index --dry-run` | Counts match `find` on the real tree |
| M2 | `embed`, `store`, `search`; resumable; one small corpus end to end | `glv search` returns sensible hits for Malta |
| M3 | TUI select / run / search screens | A user indexes Brazil without reading the README |
| M4 | Full run over the starting selection, timings in README, CI (lint + tests), tag 0.1.0 | A stranger goes from clone to first search in five minutes |
