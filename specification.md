# global-law-vector — Specification

Version 0.3, 2026-11-02. Status: milestones M1 to M3 implemented, M4 in progress (section 10).

This document records the design and the reasoning behind it. The README describes how to use
the tool; where the two differ, the code and the README are current.

## 1. Core concept

`global-law` (https://github.com/lexifina/global-law) is the law of 211 jurisdictions as one git
history: one markdown file per provision, one commit per law taking effect. Git answers exactly
*what the law said on a date*. It cannot answer *which provisions, anywhere, are about a topic*.

`global-law-vector` is a Python utility that turns a chosen subset of that tree into a vectorised
index: it reads the provision files, embeds them through a local embeddings service, and stores
text, vectors and metadata in PostgreSQL with pgvector. The result is a semantic search layer
(RAG store) over world law that any application, or an LLM through a tool, can query.

Two things make it different from "embed a folder":

1. **Selection is a first-class feature.** The tree holds 4.5 million files in 525 corpora. Nobody
   indexes all of it on the first day. Choosing countries and corpora must be trivial: a selection
   file, CLI flags, or an interactive terminal UI that shows what each choice costs.
2. **The index knows where it came from.** Every chunk is traceable to a corpus, a commit, a file
   and a position, so a search hit can be read back from `global-law` exactly, on any date.

## 2. Premises

P1. **Read-only source.** The utility never writes inside `global-law`, never runs `git checkout`
    there, and never relies on the root repository's commit hashes (they change on every rebuild).
    It relies on the submodules pinned in the root tree (`git ls-tree`), `manifest.json` as
    metadata, each corpus's pinned commit, and the files.

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
    The default model is multilingual (section 8.1); English-only models are an explicit opt-in.

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
| Corpora pinned as submodules (root tree gitlinks) | 525 |
| Corpora described in `manifest.json` | 107 |
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
┌ global-law-vector ──────────────────────────── model: arctic-embed-m-v2.0 (768) ┐
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
├── README.md                 from clone to first search
├── pyproject.toml            package `global_law_vector`, console script `glv`
├── selection.example.toml    a selection to copy; `selection.toml` itself is git-ignored
├── src/global_law_vector/
│   ├── config.py             environment and .env: paths, URLs, model, dimensions, batching
│   ├── discover.py           pinned submodules + manifest → corpora; file lists and counts
│   ├── selection.py          selection.toml read/write, resolution to corpora, cost estimate
│   ├── parse.py              front matter, title, body, citation, skip rules
│   ├── chunk.py              token-aware splitting with heading paths
│   ├── tokenizer.py          the model's tokenizer, or a heuristic when it is unavailable
│   ├── embed.py              TEI client: batching, concurrency, retry, model check
│   ├── index.py              plan + the three-thread parse/embed/write pipeline
│   ├── store.py              schema, COPY loader, upsert, delete, HNSW build, search
│   ├── search.py             query embedding + top-k with filters
│   ├── project.py            `glv project`: cached 3-D UMAP layout for the viewer
│   ├── cli.py                the `glv` commands
│   ├── sql/001_schema.sql    the schema
│   └── tui/                  Textual app: select, run, search screens
├── web/                      Next.js viewer of the index as a 3-D point cloud (read-only)
└── tests/
    ├── fixtures/             real files copied from seven corpora of different shapes
    └── test_*.py             unit tests; test_integration.py needs the services
```

Dependencies: `psycopg[binary]` only. The HTTP client is `urllib`, TOML is `tomllib`, the front
matter parser is forty lines (the corpora use flat `key: value` YAML), the CLI is `argparse`.
Optional: `tokenizers` for exact counts (else a heuristic), `textual` for the TUI,
`umap-learn` and `numpy` for `glv project`. No ML frameworks in the indexing path.

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
  indexed_at    timestamptz NOT NULL,       -- last touched
  completed_at  timestamptz                  -- NULL while a run is in progress ("partial")
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
  embedding     halfvec(768) NOT NULL,       -- dims from config (768 for arctic-embed-m-v2.0)
  UNIQUE (document_id, ordinal)
);

CREATE TABLE global_law.index_meta (         -- model id, dims, max tokens, chunker version, built_at
  key   text PRIMARY KEY,
  value text NOT NULL
);

-- after the bulk load:
CREATE INDEX chunk_embedding_hnsw ON global_law.chunk USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX chunk_document_idx   ON global_law.chunk (document_id);
CREATE INDEX document_corpus_idx  ON global_law.document (corpus_id);
```

Sizing at 768 dimensions for the whole tree: about 5 million chunks × 1.5 KB (`halfvec`) ≈ 7.5 GB
of vectors plus a comparable HNSW index; `vector` (float32) doubles both. A typical selection is
far smaller.

## 8. Pipeline and performance

### 8.1 Embedding service

In the reference setup the host runs Hugging Face text-embeddings-inference as container
`utility-embeddings`, defined in `global-law-utility/docker-compose.yml`, serving
`Snowflake/snowflake-arctic-embed-m-v2.0`.

| Property | Value |
| --- | --- |
| Endpoint on the host | `http://127.0.0.1:8080` |
| Endpoint on the Docker network `utility` | `http://utility-embeddings:80` |
| Dimensions | 768 |
| Input limit | 8,192 tokens |
| Query prefix | `query: ` (documents take no prefix) |
| Languages | multilingual; a Portuguese law sentence scored 0.68 against an English election query and 0.75 against the Portuguese one, an unrelated sentence below zero |

The utility reads the model's `/info` endpoint on start, compares it with `index_meta`, and
refuses to write into an index built with another model or dimension (P5). `TEI_URL` in config
defaults to the host endpoint.

Reachability note: the port is bound to `127.0.0.1` on the host, so a devcontainer on its own
Docker network cannot see it. Either run the utility on the host, attach the devcontainer to the
`utility` network and use `http://utility-embeddings:80`, or bind the port on all interfaces.

### 8.2 Chunking under an 8k-token limit

The input limit no longer forces splitting: 99.9% of files fit in one chunk. Splitting is kept
for retrieval quality, not for the model. Default target is 1,024 tokens per chunk, configurable,
splitting on headings then paragraphs; a whole act of 200 KB (Czechia) becomes a few dozen chunks
with its heading path, which is what a search result should land on.

### 8.3 Throughput

- Measured 2026-10-31 on the host (Intel i7-8700K, 6 cores, no GPU, TEI `cpu-latest`): about
  **1.5 chunks/s** for Malta's provisions (roughly 500 tokens each), i.e. ~750 tokens/s. The
  server's own metrics show it is CPU-bound; more client concurrency only lengthens its queue.
  At that rate Malta (66,864 files) is ~12 hours, Brazil national (14,809) ~3 hours, and the
  whole tree (4.5 M files) about 35 days: the full tree needs a GPU or a smaller model.
- Client settings that match a CPU server: 16 chunks per request, 2 concurrent requests, a
  600 s timeout. A request the client abandons is still computed by the server, so timeouts
  must be long, not short.
- The cost estimate in the TUI and in `glv select --show` uses a chunks-per-second figure
  measured on the first run and stored in `index_meta` (1.5 until then).
- Load first, build HNSW once, with `maintenance_work_mem` raised for the build. Re-indexing a
  changed corpus updates rows under the existing index.
- A rerun compares corpus SHA and file hashes and touches only new, changed or deleted files.
- Measured 2026-11-01 against a rented RTX-class GPU running TEI, from the same desktop over
  the internet: about 600 chunks/s with 32 chunks per request and 16 concurrent requests. The
  client, not the GPU, sets the pace: throughput is batch size × concurrency ÷ round-trip time.

### 8.4 Filtered search

HNSW finds its `ef_search` nearest candidates first and applies a `WHERE` clause afterwards, so
a filter on a small country returns few or no rows. Measured on a 1.6 M-chunk index of ten
countries, an unfiltered HNSW query was fine, but a single-country filter returned fewer than
ten rows in 12 of 15 searches. Search therefore scans exactly when the filter matches at most
200,000 chunks (20 ms for Malta, 1.5 s for Germany's 173,000), and otherwise uses HNSW with
`ef_search = 100` and pgvector 0.8's iterative scan.

## 9. Decisions

1. **Embedding model.** Closed 2026-10-31: `Snowflake/snowflake-arctic-embed-m-v2.0`, 768
   dimensions, multilingual (section 8.1).
2. **Vector type.** Closed: `halfvec(768)`, half the size of `vector(768)`. The model's output
   does not need 32-bit precision for ranking by cosine; `vector` remains a configuration
   switch for anyone who wants to measure the difference.
3. **Hardware for the full tree.** Closed 2026-11-01: embedding runs on a rented GPU when the
   selection is large (section 8.3; README, "Throughput and a rented GPU"). The CPU server
   remains enough for single corpora.
4. **Starting selection.** Open. Indexed so far: ten countries, 170 corpora, 1.6 M chunks.
5. **In-force date per document in v1.** Open. Costs one `git log` per file. Proposed: off by
   default, a `--with-dates` flag.
6. **Chunk target size.** Open. 1,024 tokens (section 8.2), to be checked against search quality
   on a labelled set of queries.

## 10. Milestones

| # | Deliverable | Done when | State |
| --- | --- | --- | --- |
| M0 | Specification agreed, model and starting selection chosen | Open decisions closed | model chosen; selection open |
| M1 | `discover`, `selection`, `parse`, `chunk` with fixture tests; `glv select --show` and `glv index --dry-run` | Counts match `find` on the real tree | done |
| M2 | `embed`, `store`, `search`; resumable; one small corpus end to end | `glv search` returns sensible hits for Malta | done |
| M3 | TUI select / run / search screens | A user indexes Brazil without reading the README | done |
| M4 | Full run over the starting selection, timings in README, CI (lint + tests), tag 0.1.0 | A stranger goes from clone to first search in five minutes | timings and CI done; tag pending |
