# Changelog

## 0.2.0 (2026-11-02)

**Indexes built with 0.1.0 must be rebuilt.** `glv index` refuses to write into them
(`chunker_version` 1). Index into a new schema with `GLV_SCHEMA`, or drop the old one with
`glv reset`, then run `glv index` and `glv project` again. Searching an old index still works.

### Fixed

- Chunking. The model's `tokenizer.json` truncates at 512 tokens, and counting kept that
  setting, so no count ever exceeded 512 and no document was ever split. Long acts became
  one vector, and the embeddings server silently cut anything beyond 8,192 tokens. Counting
  now ignores the file's truncation and padding.
- Paragraphs between the 1,024-token target and the model's 8,192-token limit are now split
  too, splitting keeps line breaks, and long text without spaces is split by characters.
  Chunking is linear in the size of the text; it used to count a growing string once per
  word.
- A provision that opens with its own title is no longer embedded with the title twice.
- With exact token counts, the embeddings server is asked not to truncate, so an over-long
  input is an error rather than lost text.
- Filtered search returned too few results, often none, for a country holding a small share of
  the index. Small filters are now answered by an exact scan, and wider ones by HNSW with
  iterative scans.
- `glv index --max-files` deleted stored documents beyond the limit, and marked the corpus
  complete. It does neither now.
- Ctrl-C could hang the indexer; a corpus interrupted after its last write stayed partial.
- `.env` values followed by a comment, as in `.env.example`, failed to parse.
- The viewer drew points darker and more saturated than their legend colours.

### Changed

- Planning a run no longer reads a corpus that was completed at the commit pinned now, and
  reads only the files `git diff` reports when the commit has moved.
- The index records whether token counts were exact and refuses to mix exact and estimated
  counts, as well as chunk size and document prefix.
- `corpus.chunk_count` is recorded when a corpus completes and sizes filtered searches. Two
  indexes duplicating the unique constraints are dropped by the next `glv index`.
- Front matter with block lists, block scalars and nested keys is read; invalid UTF-8 is
  logged.
- `glv status` and the TUI no longer create the schema; `glv check` hides the database
  password; known failures exit with a message instead of a traceback.
- The viewer's API takes no parameters and returns errors without details in production.
- CI runs the store against PostgreSQL with pgvector.

## 0.1.0 (2026-11-01)

First version: discovery, selection, parsing, chunking, embedding through
text-embeddings-inference, storage in PostgreSQL with pgvector, search, the `glv` command
line, the terminal interface, `glv project` and the web viewer.
