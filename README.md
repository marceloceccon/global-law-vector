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

glv check                        # tree, embeddings server and database reachable?
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
| `glv check` | Is the tree found, do the embeddings server and the database answer, and with which model. |

All settings are environment variables, see `.env.example`: tree path, database URL, embeddings URL and optional API key, model, dimensions, chunk size, batch size, workers, timeout.
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

The HNSW index is built once, at the end of a run, or with `glv index --build-index`. The
build is parallel and needs real shared memory: the pgvector service in
`global-law-utility/docker-compose.yml` sets `shm_size: 2gb` for that reason. Without it
Postgres reports "could not resize shared memory segment … No space left on device" and
`glv` retries the build without parallel workers, which is slower but works.

The index records model, dimensions, vector type and chunker version; `glv` refuses to write into
an index built with other settings. Use `GLV_SCHEMA` to keep several indexes side by side.

## Remote GPU

Embedding is the only slow stage, and it is plain HTTP, so the embeddings server can run on a
rented GPU while the files and the database stay local. On a 6-core CPU this model embeds about
1.5 chunks/s (Malta: 12 h, the whole tree: a month); on one rented RTX 4090 or L4 it is a few
hundred chunks/s, so the whole tree is a few hours and a few dollars.

1. Start a GPU pod (RunPod, Vast.ai, …) with the TEI CUDA image for its card, for example
   `ghcr.io/huggingface/text-embeddings-inference:89-latest` on an RTX 4090 (`86-latest` on
   A10/A100/RTX 30xx, `1.9` for Hopper). TEI reads its flags from environment variables, so set
   these on the pod:
   ```sh
   MODEL_ID=Snowflake/snowflake-arctic-embed-m-v2.0
   API_KEY=<a secret>              # the public URL is otherwise open to anyone
   MAX_CLIENT_BATCH_SIZE=128       # chunks per request (default 32)
   MAX_CONCURRENT_REQUESTS=4096    # the queue limit, counted in chunks (default 512)
   MAX_BATCH_TOKENS=65536          # tokens per GPU batch (default 16384)
   ```
   Expose port 80 as an HTTP port; the provider gives an HTTPS proxy URL. A Hugging Face
   Inference Endpoint running this model works the same way with its bearer token.
2. In `.env`:
   ```sh
   TEI_URL=https://<pod-id>-80.proxy.runpod.net   # or the endpoint URL
   TEI_API_KEY=<the secret>
   GLV_BATCH_SIZE=128     # match MAX_CLIENT_BATCH_SIZE; 32 if the pod keeps its default
   GLV_WORKERS=16
   ```
   Over the internet each request costs a round trip of a few hundred milliseconds whatever
   its size, so throughput is chunks-per-request × concurrent requests ÷ latency. The GPU is
   idle most of the time; more GPUs do not help, larger requests do.
3. `glv check`, then `glv index`. The run resumes anything a CPU run already stored, because
   the model, dimensions and index settings are the same.

Measured on 2026-11-01 from a devcontainer in Brazil to a RunPod RTX-class pod with the
default limits (32 per request, 16 workers): about 600 chunks/s, versus 1.5 chunks/s on the
local 6-core CPU. Parsing, embedding and writing run in three threads so the local work
overlaps the network. The client sends its own
`User-Agent`, because the Cloudflare proxy in front of RunPod rejects Python's default one.

The whole tree sends about 5.5 GB of text up and receives about 14 GB of vectors; stop the pod
when `glv status` shows every corpus indexed.

## Development

```sh
pip install -e '.[dev]'
ruff check src tests
pytest                 # unit tests on fixtures from seven real corpora
pytest tests/test_integration.py   # end to end; runs only when both services are reachable
```

Licence: CC0 1.0, like global-law.
