# global-law-vector

Semantic search over the law of the world. `glv` reads a chosen subset of
[global-law](https://github.com/lexifina/global-law), the law of 211 jurisdictions as one git
history, embeds every provision with a local embeddings server, and stores text, vectors and
metadata in PostgreSQL + [pgvector](https://github.com/pgvector/pgvector). Then it answers
"which provisions, anywhere, are about X" in any language.

Status: milestones M1 and M2 of [specification.md](specification.md) are implemented (discover,
select, parse, chunk, embed, store, search, CLI) plus a first TUI. See the specification for the
core concept, the information flow and the premises.

## Five minutes to the first search

Requirements: Python 3.11+, a clone of global-law with its submodules, and the two containers
from the sibling project `global-law-utility` (text-embeddings-inference on port 8080 and
pgvector on 5432):

```sh
git clone --recurse-submodules https://github.com/lexifina/global-law
cd global-law-utility && docker compose up -d && cd ..

git clone <this repo> global-law-vector && cd global-law-vector
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'          # or `pip install -e .` for the bare CLI
cp .env.example .env             # defaults match the containers; edit if needed

glv list                         # the 525 pinned corpora, 213 countries
glv select --country malta --corpus brazil/national/acts --show
glv index --dry-run              # parse and chunk, embed nothing
glv index                        # embed and store; stop and rerun at will
glv search "prazo de prescrição" --country brazil
glv tui                          # or do all of the above interactively
```

Every value in `.env` has a default that matches the containers on localhost. The global-law
tree is found at `GLOBAL_LAW_PATH`, else at `./global-law` or `../global-law`.

## Commands

| Command | Does |
| --- | --- |
| `glv list [--country X]` | Every corpus in the tree, marking the selected ones. |
| `glv select --country X --corpus c/l/n --exclude … --all --clear --show` | Edit `selection.toml` and show its cost: files, bytes, estimated chunks and time. |
| `glv index [--dry-run] [--country …] [--max-files N]` | Index the selection (or a one-off selection). Resumable: a rerun indexes only new, changed or deleted files. |
| `glv status [--all]` | Per corpus: not indexed, partial, stale (corpus commit changed) or indexed. |
| `glv search "question" [--country X] [--top N]` | Cosine search; prints score, path, title, citation, heading and a snippet. |
| `glv reset` | Drop the schema (asks first). |
| `glv tui` | Select with live totals, run with progress, search with detail. |

## How it works

```
discover → select → plan → parse → chunk → embed (TEI) → store (pgvector) → search
```

- **Discover** reads the submodules pinned in the root tree (`git ls-tree`) and the manifest for
  metadata. Nothing is ever written in the tree.
- **Select** resolves countries and corpora to a list of corpus paths and prices it with cached
  per-corpus file counts.
- **Plan** compares each corpus on disk with the database by corpus commit and file hash.
- **Parse** handles front matter, headings, bare text, and skips READMEs and the like.
- **Chunk** keeps a provision whole when it fits the target (1,024 tokens by default) and
  otherwise splits on headings then paragraphs, keeping the heading path.
- **Embed** sends batches to text-embeddings-inference with retries, on a thread pool.
- **Store** writes each group of documents in one transaction through COPY, and builds the HNSW
  index once at the end.
- **Search** prefixes the question with `query: `, embeds it and ranks chunks by cosine.

The index records model, dimensions, vector type and chunker version; `glv` refuses to write into
an index built with other settings. Use `GLV_SCHEMA` to keep several indexes side by side.

## Development

```sh
pip install -e '.[dev]'
ruff check src tests
pytest                 # unit tests on fixtures from seven real corpora
pytest tests/test_integration.py   # end to end; runs only when both services are reachable
```

Licence: CC0 1.0, like global-law.
