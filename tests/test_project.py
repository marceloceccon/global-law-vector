"""The projection step without UMAP or a database: allocation, normalisation, file format."""

from __future__ import annotations

import json

import pytest

from global_law_vector.project import SamplePoint, UmapParams, allocate, write_projection

np = pytest.importorskip("numpy")


def test_allocate_balances_small_countries_and_fills_from_large_ones():
    by_country = {
        "malta": list(range(100)),
        "france": list(range(1000, 11000)),
        "spain": list(range(20000, 25000)),
    }
    chosen = allocate(by_country, 3000)
    counts = {}
    for _, c in chosen:
        counts[c] = counts.get(c, 0) + 1
    assert len(chosen) == 3000
    assert counts["malta"] == 100  # everything it has
    assert counts["france"] == counts["spain"] == 1450  # the rest split evenly
    # shuffled: the first 300 are not a single country
    assert len({c for _, c in chosen[:300]}) == 3
    # deterministic
    assert allocate(by_country, 3000) == chosen


def test_allocate_when_total_is_smaller_than_n():
    chosen = allocate({"a": [1, 2], "b": [3]}, 10)
    assert sorted(cid for cid, _ in chosen) == [1, 2, 3]


def test_normalise_and_write_round_trip(tmp_path):
    from global_law_vector.project import normalise

    rng = np.random.default_rng(0)
    xyz = normalise(rng.normal(size=(500, 3)) * 5 + 100)
    radii = np.linalg.norm(xyz, axis=1)
    assert abs(np.percentile(radii, 99) - 1) < 1e-5
    assert np.all(np.abs(np.median(xyz, axis=0)) < 1e-5)

    points = [SamplePoint(i, "brazil" if i % 2 else "malta", f"c{i % 3}") for i in range(500)]
    out = tmp_path / "deep" / "projection.json"
    doc = write_projection(
        out, points, xyz, {"brazil": 1000, "malta": 20}, UmapParams(), 300, "model-x", 768
    )
    saved = json.loads(out.read_text())
    assert saved == doc
    assert saved["method"] == "umap" and saved["params"]["fit"] == 300
    assert saved["sample"] == 500 and len(saved["xyz"]) == 1500
    assert saved["countries"] == ["brazil", "malta"] and saved["corpora"] == ["c0", "c1", "c2"]
    assert saved["country"][1] == 0 and saved["country"][0] == 1
    assert saved["total_chunks"] == 1020
    assert not out.with_suffix(".json.tmp").exists()
