"""Stage 8, SEARCH: embed a question with the query prefix and rank chunks by cosine."""

from __future__ import annotations

from collections.abc import Sequence

from .embed import TEIClient
from .store import Hit, Store


def search(
    store: Store,
    client: TEIClient,
    query: str,
    *,
    query_prefix: str = "query: ",
    top: int = 10,
    countries: Sequence[str] = (),
    corpora: Sequence[str] = (),
) -> list[Hit]:
    text = query.strip()
    if not text:
        return []
    if query_prefix and not text.startswith(query_prefix):
        text = query_prefix + text
    [vector] = client.embed([text])
    return store.search(vector, top=top, countries=countries, corpora=corpora)
