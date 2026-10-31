# global-law-vector

A Python utility that turns a chosen subset of [global-law](https://github.com/lexifina/global-law),
the law of 211 jurisdictions as one git history, into a vectorised search index in PostgreSQL +
pgvector, using a local embeddings server.

Status: specification stage. Read [specification.md](specification.md) for the core concept, the
information flow, the premises, the selection model (file, CLI and terminal UI) and the schema.

Infrastructure (text-embeddings-inference + pgvector in Docker) lives in the sibling project
`global-law-utility` and is shared with other tools on the same host.
