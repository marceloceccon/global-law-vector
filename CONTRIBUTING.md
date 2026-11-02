# Contributing

Bug reports and pull requests are welcome. For anything larger than a fix, open an issue first
so the approach can be agreed before you spend time on it.

## Setting up

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
cd web && npm install
```

The unit tests need neither the embeddings server nor the database. The integration test and
the viewer do; the README's Requirements section starts both with Docker.

## Before opening a pull request

```sh
ruff check src tests
ruff format src tests
pytest
cd web && npm run typecheck && npm run build
```

CI runs the same commands. Add a test with every bug fix, one that fails without the fix.
Test fixtures are real files from global-law corpora, so add one from the corpus that showed
the problem rather than writing a synthetic one.

## Things to keep in mind

- **The index records its settings.** Anything that changes the vectors written for the same
  input (the model, the text sent to it, the chunking rules) must either be recorded in
  `index_meta` and checked in `Store.ensure_meta`, or bump `CHUNKER_VERSION` in `chunk.py`.
  An index must never mix two ways of embedding.
- **global-law is read-only.** No command writes to the tree or runs `git checkout` in it.
- **Few dependencies.** The indexer depends on `psycopg` alone; everything else is optional and
  imported only where it is needed. Please keep it that way.
- **Commit messages** are in English and in the imperative mood ("Fix …", "Add …").

## Licence

By contributing you agree that your contribution is released under
[CC0 1.0](LICENSE), like the rest of the project.
