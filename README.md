# global-law-vector

Semantic search over the law of the world. `glv` reads a chosen part of
[global-law](https://github.com/lexifina/global-law), which holds the law of 211 jurisdictions
as one git history, and embeds every provision with a multilingual model. It stores the text,
the vectors and their provenance in PostgreSQL with
[pgvector](https://github.com/pgvector/pgvector). The result answers "which provisions,
anywhere, are about X", in any language, and every hit can be read back from global-law at the
exact commit it was indexed from.

Comparing how countries regulate the same thing is slow work: each country has its own
language, terminology and structure. Asked in English, without naming a country, the index
finds the reporting duties of Argentina's anti-money-laundering law, Germany's Money Laundering
Act and a French ministerial order side by side:

```
$ glv search "obligation to report suspicious transactions to the financial intelligence unit" --top 8 --short
 1. 0.624  argentina/national/acts/ley-25246/art-017.md
    Art. 17
 2. 0.580  argentina/national/acts/ley-25246/art-021.md
    Art. 21
 3. 0.569  argentina/national/acts/ley-25246/art-014.md
    Art. 14
 4. 0.554  germany/bund/acts/gwg_2017/0030.md
    § 30 Analyse von Meldungen und Informationen
 5. 0.553  argentina/chaco/acts/ley-1529-b/art-058.md
    ley-1529-b / art-058
 6. 0.552  france/national/acts/textes/arrete/2004/arrete-2004-11-12-441606/315-2.md
    Article 315-2
 7. 0.534  argentina/national/acts/ley-25246/art-019.md
    Art. 19
 8. 0.531  germany/bund/acts/gwg_2017/0043.md
    § 43 Meldepflicht von Verpflichteten, Verordnungsermächtigung
```

The index is plain PostgreSQL, so the same search is a SQL query away from any application or
from a language model's retrieval tool.

![A 40,000-chunk sample of the index drawn by the web viewer, coloured by country](docs/images/viewer-by-country.png)

*The web viewer in [`web/`](web/README.md) shows a country-balanced sample of 40,000 chunks
from a 1.6-million-chunk index of ten countries. The 768-dimensional embeddings are laid out
in three dimensions with UMAP and coloured by country.*

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Configuration](#configuration)
- [The index](#the-index)
- [Querying the index from your own code](#querying-the-index-from-your-own-code)
- [How indexing works](#how-indexing-works)
- [How search works](#how-search-works)
- [Throughput and a rented GPU](#throughput-and-a-rented-gpu)
- [Running in a container](#running-in-a-container)
- [The web viewer](#the-web-viewer)
- [Development](#development)
- [Licence](#licence)

## Requirements

- Python 3.11 or later.
- A clone of global-law with its submodules. A full checkout is about 21 GB, but only the
  corpora you select are ever read.
- A [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
  (TEI) server running `Snowflake/snowflake-arctic-embed-m-v2.0`, the default model.
- PostgreSQL with the pgvector extension, version 0.8 or later for the best filtered search
  (see [How search works](#how-search-works)).

The two services can run anywhere `glv` can reach over HTTP and the PostgreSQL protocol. On one
machine with Docker:

```sh
docker run -d --name glv-embeddings -p 127.0.0.1:8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id Snowflake/snowflake-arctic-embed-m-v2.0

docker run -d --name glv-pgvector -p 127.0.0.1:5432:5432 --shm-size=2g \
  -e POSTGRES_USER=utility -e POSTGRES_PASSWORD=utility -e POSTGRES_DB=utility \
  pgvector/pgvector:pg16
```

These match the defaults in `.env.example`, so nothing else needs configuring. Pick your own
password for anything other than a local machine. The `--shm-size` matters for the HNSW index
build; see [Building the HNSW index](#building-the-hnsw-index).

## Quick start

```sh
git clone --recurse-submodules https://github.com/lexifina/global-law
git clone https://github.com/marceloceccon/global-law-vector
cd global-law-vector

python -m venv .venv && . .venv/bin/activate
pip install -e '.[tokenizer]'    # the bare CLI, plus the model's tokenizer for exact counts
cp .env.example .env             # defaults match the containers above

glv check                        # tree, embeddings server and database: found and answering?
glv list --country malta         # what the tree holds for a country
glv select --country malta --show   # choose, and see files, size and estimated time
glv index --dry-run              # parse and chunk the selection, embed nothing
glv index                        # embed and store; stop with Ctrl-C and rerun at will
glv search "obligations of a landlord when the rent is increased" --country malta
```

A search prints the score, the file in global-law, its title and citation, the heading path
inside the provision and a snippet:

```
 1. 0.515  malta/national/acts/cap/116/5.md
    5. Renewals of tenancy and new tenancies.
    # 5. Renewals of tenancy and new tenancies. (1) When a landlord intends to let or to renew the letting of - (a) a dwelling house at a rent higher than or on con…
 2. 0.481  malta/national/acts/cap/116/7.md
    7. Tenancy agreements with effect from 1st April, 1939, or from later date, made before enactment of this Ordinance.
    …
```

`glv tui` does the same interactively. It offers a selection screen with live cost totals, a
run screen with progress and a search screen.

## Commands

| Command | What it does |
| --- | --- |
| `glv check` | Reports where the tree is and whether the embeddings server and the database answer, with which model and dimensions. |
| `glv list [--country X]` | Lists every corpus in the tree, marking the selected ones. |
| `glv select [--country X] [--corpus c/l/n] [--exclude X] [--all] [--clear] [--show]` | Edits `selection.toml` and shows what the selection costs in files, bytes, chunks and time. |
| `glv index [--dry-run] [--country X] [--max-files N] [--no-build-index]` | Indexes the selection, or a one-off selection given by flags. It is resumable: a rerun touches only new, changed or deleted files. |
| `glv status [--all]` | Shows each corpus as not indexed, partial (a run stopped), stale (the corpus has a newer commit) or indexed. |
| `glv search "question" [--country X] [--corpus c/l/n] [--top N]` | Runs a semantic search and prints score, path, title, citation, heading and snippet. |
| `glv reset` | Drops the index schema after asking for confirmation. |
| `glv project [--points 40000] [--fit 20000]` | Caches a 3-D UMAP layout of a sample of the index for the web viewer. It needs the `project` extra. |
| `glv tui` | Opens the terminal interface. It needs the `tui` extra. |

`--country` and `--corpus` can be repeated. Country names and corpus paths are those of the
global-law tree, for example `brazil` or `brazil/national/acts`. Exit codes are 0 on success,
1 on a failure such as an unreachable service or a mismatched index, 2 on a usage error such as
an unknown country, and 130 when interrupted.

Optional extras install with pip: `tokenizer` (exact token counts), `tui` (Textual), `project`
(UMAP and NumPy for `glv project`) and `dev` (tests and lint).

## Configuration

All settings are environment variables. `glv` also reads a `.env` file in the working
directory, which never overrides the real environment. `.env.example` lists them with their
defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GLOBAL_LAW_PATH` | `./global-law`, `../global-law` or `../../global-law` | Root of the global-law checkout. |
| `DATABASE_URL` | `postgresql://utility:utility@127.0.0.1:5432/utility` | PostgreSQL connection string. |
| `GLV_SCHEMA` | `global_law` | Schema holding the index. Use several to keep indexes side by side. |
| `TEI_URL` | `http://127.0.0.1:8080` | Embeddings server. |
| `TEI_API_KEY` | unset | Sent as `Authorization: Bearer …` for a remote server. |
| `GLV_MODEL` | `Snowflake/snowflake-arctic-embed-m-v2.0` | Model the server must be running. |
| `GLV_DIMS` | `768` | Embedding width. |
| `GLV_MAX_TOKENS` | `8192` | The model's input limit; no chunk exceeds it. |
| `GLV_QUERY_PREFIX` | `query: ` | Prepended to search queries, as the model expects. |
| `GLV_DOCUMENT_PREFIX` | empty | Prepended to indexed text. The default model uses none. |
| `GLV_VECTOR_TYPE` | `halfvec` | `halfvec` (16-bit) or `vector` (32-bit, twice the size). |
| `GLV_CHUNK_TOKENS` | `1024` | Target chunk size in tokens. |
| `GLV_BATCH_SIZE` | `16` | Chunks per embeddings request. TEI accepts 32 by default. |
| `GLV_WORKERS` | `2` | Concurrent embeddings requests. |
| `GLV_TEI_TIMEOUT` | `600` | Seconds to wait for one request. Keep it long on a CPU server. |
| `GLV_TOKENIZER` | unset | Path to a `tokenizer.json`. Otherwise it is fetched from the Hugging Face hub once and cached. |
| `HF_TOKEN` | unset | Only for that first tokenizer download, if the hub asks for it. |
| `GLV_SELECTION` | `selection.toml` | Selection file. |
| `GLV_CACHE_DIR` | `~/.cache/global-law-vector` | Tokenizer copy, corpus statistics and the projection file. |
| `GLV_CHUNKS_PER_SECOND` | `1.5` | Throughput used for time estimates until a run has measured one. |

The model, dimensions, vector type, chunker version, chunk size and document prefix are
recorded in the index when it is created. `glv index` refuses to write into an index built with
different values, because mixing them would silently corrupt the results. To change one,
choose another `GLV_SCHEMA` or run `glv reset`.

## The index

One schema, `global_law` by default, holds four tables:

| Table | One row per | Main columns |
| --- | --- | --- |
| `corpus` | indexed corpus | `path` (`germany/bund/acts`), `country`, `language`, `commit_sha` (the submodule commit indexed), `completed_at` (null while a run is in progress), `chunk_count` |
| `document` | provision file | `path` (relative to the global-law root), `title`, `citation`, `front_matter` (jsonb), `content_sha` (sha256 of the file), `in_force_date` (reserved for indexing past versions; not filled yet) |
| `chunk` | piece of a provision | `ordinal`, `heading_path` (`Chapter II > § 4`), `text`, `token_count`, `embedding` (`halfvec(768)`) |
| `index_meta` | setting | `model`, `dims`, `vector_type`, `chunker_version`, `chunk_tokens`, `document_prefix`, `token_counting`, `chunks_per_second` |

The full definition is in [`src/global_law_vector/sql/001_schema.sql`](src/global_law_vector/sql/001_schema.sql).
`chunk.embedding` has an HNSW index with cosine distance (`chunk_embedding_hnsw`).

**What is embedded.** Each chunk is embedded as its document's title and heading path, a blank
line, then the chunk text. A bare article such as São Paulo's `art-0001.md` thereby carries
the act it belongs to. `chunk.text` stores the text alone.

**Provenance.** `corpus.commit_sha` and `document.path` locate every hit in global-law exactly.
`git -C global-law/<corpus path> show <commit_sha>:<path inside the corpus>` prints the
provision as it was indexed, even after the corpus has moved on.

**Size.** At 768 dimensions a `halfvec` takes 1.5 KB per chunk, and the HNSW index is of a
similar size. The ten-country index behind the screenshot holds 1.6 million chunks. The whole
tree has 4.5 million files and would give several times as many chunks, because corpora that
store a whole act per file split heavily: Czechia's first 3,000 files gave 19 chunks each.

## Querying the index from your own code

The index is plain PostgreSQL, so any client can search it. There are two rules. Embed the
question with the same model, prefixed with `query: `, and compare with cosine distance (`<=>`).

From Python, with this package installed:

```python
from global_law_vector.config import Config
from global_law_vector.embed import TEIClient
from global_law_vector.search import search
from global_law_vector.store import Store

cfg = Config.from_env()
with Store(cfg.database_url, cfg.schema, cfg.vector_type, cfg.dims) as store:
    client = TEIClient(cfg.tei_url, api_key=cfg.tei_api_key)
    for hit in search(store, client, "prazo de prescrição", countries=["brazil"], top=5):
        print(f"{hit.score:.3f}", hit.path, hit.heading_path)
```

From SQL, given the query embedding as a pgvector literal `$1`. The literal comes from
`POST /embed` on the TEI server with `{"inputs": "query: …"}`.

```sql
SET hnsw.ef_search = 100;                -- candidate list; pgvector's default is 40
SET hnsw.iterative_scan = relaxed_order; -- pgvector 0.8+: keep scanning when a filter rejects rows

SELECT 1 - (k.embedding <=> $1::halfvec(768)) AS score,
       c.path AS corpus, c.commit_sha, d.path, d.title, d.citation, k.heading_path, k.text
FROM global_law.chunk k
JOIN global_law.document d ON d.id = k.document_id
JOIN global_law.corpus   c ON c.id = d.corpus_id
WHERE c.country = 'brazil'               -- optional
ORDER BY k.embedding <=> $1::halfvec(768)
LIMIT 10;
```

Read [How search works](#how-search-works) before relying on a filtered query like this one
against a large index.

## How indexing works

```
discover → select → plan → parse → chunk → embed (TEI) → store (pgvector)
```

- **Discover** lists the corpora pinned as submodules in the global-law tree (`git ls-tree`),
  with metadata from its `manifest.json`. Nothing in the tree is ever written to.
- **Select** turns countries and corpora into a list of corpus paths and prices it from cached
  file counts.
- **Plan** compares each corpus with the database. A corpus completed at the commit pinned
  now is skipped without reading a file. If its commit moved, `git diff` between the two
  commits names the changed files, and only those are read and hashed. After an interrupted
  run, or where git cannot answer, every file is hashed. Only new and changed files are
  indexed, and only documents whose files are gone are deleted.
- **Parse** reads front matter, title, citation and body, and skips files such as `README.md`
  that are not law.
- **Chunk** keeps a provision whole when it fits the 1,024-token target, counted with the
  model's own tokenizer. A longer one is split on its headings, then its paragraphs, then
  sentences and words, and each chunk keeps its heading path. No chunk exceeds the target, and
  chunks do not overlap: statutes are split along their own structure. How much splitting
  happens depends on how a corpus stores its law. Measured on 2026-11-02 by parsing and
  chunking every file with the model's tokenizer, nothing embedded:

  | Corpus | Files | Chunks | Files split | Files above the model's 8,192-token limit | Median file |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Malta, one provision per file | 66,864 | 75,650 | 4.6 % | 251 | 141 tokens |
  | Czechia, one act per file (first 3,000 files) | 3,000 | 56,104 | 58 % | 471 | 1,402 tokens |
  | Germany, one section per file | 113,478 | 136,096 | 5.2 % | 279 | 166 tokens |
  | Brazil, one article per file | 14,809 | 15,010 | 1.0 % | 0 | 86 tokens |

- **Embed** sends batches to TEI on a thread pool, with retries and back-off.
- **Store** writes each group of documents and their chunks in one transaction, chunks through
  `COPY`.

Parsing, embedding and writing run in three threads, so the embeddings server never waits for
the disk or the database. A run can be stopped at any moment, and a rerun continues from what
was stored. A corpus counts as indexed only when every file of it has been written.

### Building the HNSW index

The HNSW index is built once, after the first run, because building it before a bulk load is
far slower. Later runs update rows under the existing index. `--no-build-index` defers the
build when several runs follow each other.

A changed provision is replaced by deleting its rows and inserting new ones. After a refresh
that replaced a large share of a corpus, `VACUUM (ANALYZE)` on the `chunk` and `document`
tables reclaims the dead rows and repairs the HNSW graph around them. If recall ever seems to
have degraded after many refreshes, `REINDEX INDEX CONCURRENTLY global_law.chunk_embedding_hnsw`
rebuilds the graph without blocking searches.

A parallel build keeps its graph in shared memory. Docker gives a container 64 MB of `/dev/shm`
unless told otherwise, and the build then fails with "could not resize shared memory segment …
No space left on device". `glv` retries without parallel workers, which works but is a few
times slower. Starting PostgreSQL with `--shm-size=2g` (or `shm_size: 2gb` in Compose) avoids
the retry.

## How search works

`glv search` prefixes the question with `query: `, embeds it and ranks chunks by cosine
similarity. Without a filter it uses the HNSW index with a candidate list of 100, which takes
about 10 ms on the 1.6-million-chunk index. In the measurements below, HNSW found at least nine
of the true top ten for every unfiltered query.

Filters need more care, because HNSW applies them after finding its candidates. Ask a plain
HNSW query for the top ten Maltese chunks in an index where Malta holds 0.1 % of the rows, and
the index returns its 40 nearest candidates from the whole world. The filter then leaves one
or none of them. `glv` avoids this in two ways:

- A filter matching at most 200,000 chunks is answered by an exact scan of those chunks. That
  takes about 20 ms for Malta, 0.15 s for Portugal and 1.5 s for Germany's 173,000 chunks, and
  always returns the true top results.
- A wider filter uses the HNSW index with pgvector 0.8's iterative scan, which keeps searching
  until enough rows pass the filter.

The size of a filter comes from the chunk count each corpus records when its run completes, so
choosing between the two costs one small query.

Measured on that index with three queries, each run with no filter and with five single-country
filters, the top ten matched an exact scan in 16 of the 18 searches and shared nine of ten
results in the other two. Plain HNSW returned fewer than ten results in 12 of the 15 filtered
searches, and none at all in 10 of them.

## Throughput and a rented GPU

Embedding is the only slow stage. The default model on a six-core desktop CPU embeds about 1.5
chunks per second, which is about 14 hours for Malta's 75,650 chunks and over a month for
the whole tree.

Because the embeddings server speaks plain HTTP, it can run on a rented GPU while the files and
the database stay local:

1. Start a GPU instance with the TEI image built for its card, for example
   `ghcr.io/huggingface/text-embeddings-inference:89-latest` on an RTX 4090 or `86-latest` on
   an A10, A100 or RTX 30xx. The TEI documentation lists the images for other GPUs. TEI reads
   its options from environment variables:
   ```sh
   MODEL_ID=Snowflake/snowflake-arctic-embed-m-v2.0
   API_KEY=<a long random secret>   # without it the public URL is open to anyone
   MAX_CLIENT_BATCH_SIZE=128        # chunks per request (default 32)
   MAX_CONCURRENT_REQUESTS=4096     # queue limit, counted in chunks (default 512)
   MAX_BATCH_TOKENS=65536           # tokens per GPU batch (default 16384)
   ```
   Expose port 80 over HTTPS. A Hugging Face Inference Endpoint serving the same model works
   the same way with its token.
2. In `.env`:
   ```sh
   TEI_URL=https://<your instance>
   TEI_API_KEY=<the secret>
   GLV_BATCH_SIZE=128     # match MAX_CLIENT_BATCH_SIZE; 32 if the server keeps its default
   GLV_WORKERS=16
   ```
   Over the internet every request costs a round trip of a few hundred milliseconds, so
   throughput is chunks per request times concurrent requests divided by latency. Larger and
   more concurrent requests help. More GPUs do not.
3. Run `glv check`, then `glv index`. A run resumes whatever a CPU run already stored, since the
   model and settings are the same.

Measured from a desktop in Brazil to a rented RTX-class GPU with TEI's default limits (32 chunks
per request, 16 workers), throughput was about 600 chunks per second, 400 times the local CPU.
The whole tree sends
about 5.5 GB of text and receives about 14 GB of vectors. Stop the instance when `glv status`
shows everything indexed.

Some hosted proxies reject Python's default `User-Agent`, so the client sends its own.

## Running in a container

From inside a container, `127.0.0.1` is the container itself, not the host, so the defaults do
not reach services published on the host. Put the container on the same Docker network as the
services and address them by name:

```sh
docker network create glv
docker network connect glv glv-embeddings
docker network connect glv glv-pgvector
docker network connect glv <your container>
```

Then set `TEI_URL=http://glv-embeddings:80` and
`DATABASE_URL=postgresql://utility:utility@glv-pgvector:5432/utility` in `.env`.

## The web viewer

[`web/`](web/README.md) is a small Next.js application that draws a sample of the index as a
3-D point cloud you can orbit. Points are coloured by country, or by the nearest of eight legal
domains, and both can be filtered. It only reads from the database.

```sh
pip install -e '.[project]'
glv project                      # UMAP layout of 40,000 chunks: about two minutes, cached
cd web && npm install && npm run dev   # http://localhost:3000
```

Its own [README](web/README.md) covers the settings, serving it on a network and how the
picture is computed.

## Development

```sh
pip install -e '.[dev]'
ruff check src tests && ruff format --check src tests
pytest                               # unit tests on files taken from seven real corpora
```

The integration test in `tests/test_integration.py` runs end to end against a real TEI
server and database, and skips itself when they are not reachable. CI runs lint, format and
tests on Python 3.11 and 3.13, and a typecheck and production build of the web viewer.

[`specification.md`](specification.md) records the design: premises, information flow, data
model and the decisions behind them.

## Licence

The code is released under [CC0 1.0](LICENSE): no rights reserved.

That covers this repository only. global-law's own repository is also CC0, but the law text
lives in each country's repository under that repository's terms. An index built with `glv`
contains that text in `chunk.text`, so publishing an index dump, a hosted search or the text
of results is subject to the terms of the corpora it holds.

The default embedding model,
[Snowflake/snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0),
is published by Snowflake under the Apache 2.0 licence. `glv` calls it through an embeddings
server and does not redistribute it.
