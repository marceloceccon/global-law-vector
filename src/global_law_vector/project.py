"""`glv project`: a cached 3-D layout of a sample of the index, for the web viewer.

The viewer in web/ draws tens of thousands of chunks as points. A linear projection (PCA)
keeps the global shape of the embedding space but squashes local structure, so the cloud
looks like one blob. UMAP keeps neighbourhoods instead, at a cost of minutes of CPU, which is
why it runs here, once, and is cached as JSON that the web API serves.

Method, as written into the file: a country-balanced, seeded sample of `points` chunks (every
country gets an equal share; unused share flows to the larger ones); UMAP with three
components fitted on the first `fit` of them (the order is shuffled, so the fit sample is
balanced too) and the rest placed with `transform`; coordinates centred and scaled so the
99th-percentile radius is 1. Defaults: 40,000 points, fit on 20,000, n_neighbors=20,
min_dist=0.1, metric=cosine, seed 42. Nothing is embedded or written to the database.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .store import Store

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
Progress = Callable[[str], None]


@dataclass(slots=True)
class UmapParams:
    n_neighbors: int = 20
    min_dist: float = 0.1
    metric: str = "cosine"
    seed: int = 42


@dataclass(slots=True)
class SamplePoint:
    id: int
    country: str
    corpus: str


def allocate(by_country: dict[str, list[int]], n: int, seed: int = 42) -> list[tuple[int, str]]:
    """Water-filling: each country gets min(its size, an equal share of what is left), smallest
    first, so unused share flows to the larger ones. The result is shuffled so any prefix is as
    balanced as the whole."""
    names = sorted(by_country, key=lambda k: (len(by_country[k]), k))
    remaining = n
    out: list[tuple[int, str]] = []
    for i, name in enumerate(names):
        share = remaining // (len(names) - i)
        take = min(share, len(by_country[name]))
        out.extend((cid, name) for cid in by_country[name][:take])
        remaining -= take
    random.Random(seed).shuffle(out)
    return out


def sample_points(store: Store, n: int, seed: int = 42) -> tuple[list[SamplePoint], dict[str, int]]:
    """A country-balanced sample of chunk ids with their country and corpus, plus the number
    of chunks each country has in the index. The same seed gives the same sample from an
    unchanged index."""
    by_country: dict[str, list[int]] = {}
    meta: dict[int, tuple[str, str]] = {}
    totals: dict[str, int] = {}
    for cid, country, corpus, total in store.sample_by_country(n, seed):
        by_country.setdefault(country, []).append(cid)
        meta[cid] = (country, corpus)
        totals[country] = total
    chosen = allocate(by_country, n, seed)
    return [SamplePoint(cid, *meta[cid]) for cid, _ in chosen], totals


def fetch_embeddings(
    store: Store, ids: list[int], batch: int = 2000, progress: Progress | None = None
) -> np.ndarray:
    """The vectors for `ids`, in that order, as float32."""
    import numpy as np

    out = np.empty((len(ids), store.dims), dtype=np.float32)
    pos = {cid: i for i, cid in enumerate(ids)}
    seen = 0
    for rows in store.iter_embeddings(ids, batch):
        for cid, text in rows:
            vec = np.asarray(json.loads(text), dtype=np.float32)
            if vec.shape != (store.dims,):
                raise RuntimeError(
                    f"chunk {cid} has {vec.size} dimensions, GLV_DIMS is {store.dims}"
                )
            out[pos[cid]] = vec
            seen += 1
        if progress:
            progress(f"fetched {seen:,}/{len(ids):,} embeddings")
    if seen != len(ids):
        raise RuntimeError(f"{len(ids) - seen} sampled chunks vanished while projecting; rerun")
    return out


def reduce_umap(
    x: np.ndarray, fit: int, params: UmapParams, progress: Progress | None = None
) -> np.ndarray:
    """3-D UMAP: fit on the first `fit` rows, place the rest with transform()."""
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "UMAP is not installed: pip install 'global-law-vector[project]'"
        ) from exc

    fit = min(max(fit, 1), len(x))
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        random_state=params.seed,
        verbose=False,
    )
    t0 = time.monotonic()
    if progress:
        progress(f"fitting UMAP on {fit:,} points (this is the slow part)")
    head = reducer.fit_transform(x[:fit])
    if fit < len(x):
        if progress:
            progress(f"fitted in {time.monotonic() - t0:.0f}s; placing {len(x) - fit:,} more")
        tail = reducer.transform(x[fit:])
        import numpy as np

        return np.vstack([head, tail]).astype(np.float32)
    return head.astype("float32")


def normalise(xyz: np.ndarray) -> np.ndarray:
    """Centre on the median and scale so the 99th-percentile radius is 1."""
    import numpy as np

    centred = xyz - np.median(xyz, axis=0)
    radius = np.percentile(np.linalg.norm(centred, axis=1), 99) or 1.0
    return (centred / radius).astype(np.float32)


def write_projection(
    path: Path,
    points: list[SamplePoint],
    xyz: np.ndarray,
    totals: dict[str, int],
    params: UmapParams,
    fit: int,
    model: str | None,
    dims: int,
) -> dict[str, Any]:
    """Columnar JSON the web API reads. Written atomically."""
    countries = sorted({p.country for p in points})
    corpora = sorted({p.corpus for p in points})
    ci = {c: i for i, c in enumerate(countries)}
    ki = {c: i for i, c in enumerate(corpora)}
    doc: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "method": "umap",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "dims": dims,
        "params": {**asdict(params), "fit": min(fit, len(points))},
        "sample": len(points),
        "total_chunks": sum(totals.values()),
        "country_totals": totals,
        "countries": countries,
        "corpora": corpora,
        "ids": [p.id for p in points],
        "country": [ci[p.country] for p in points],
        "corpus": [ki[p.corpus] for p in points],
        "xyz": [round(float(v), 4) for v in xyz.reshape(-1)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    return doc


def project(
    store: Store,
    out: Path,
    points: int = 40_000,
    fit: int = 20_000,
    params: UmapParams | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Sample, fetch, reduce, write. Returns the document written."""
    params = params or UmapParams()
    say = progress or (lambda m: None)
    say(f"sampling {points:,} chunks balanced across countries")
    sample, totals = sample_points(store, points, params.seed)
    if len(sample) < 10:
        raise RuntimeError("the index holds too few chunks to project")
    x = fetch_embeddings(store, [p.id for p in sample], progress=say)
    xyz = normalise(reduce_umap(x, fit, params, say))
    meta = store.get_meta()
    doc = write_projection(out, sample, xyz, totals, params, fit, meta.get("model"), store.dims)
    say(f"wrote {out} ({len(sample):,} points, {len(totals)} countries)")
    return doc
