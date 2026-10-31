from __future__ import annotations

from pathlib import Path

import pytest

from global_law_vector.discover import CorpusStats
from global_law_vector.selection import Selection, estimate, human_bytes, human_duration


def test_resolve_countries_and_corpora(tree):
    sel = Selection(countries=["brazil"], corpora=["germany/bund/acts"])
    assert [c.path for c in sel.resolve(tree)] == [
        "brazil/national/acts", "brazil/sao-paulo/acts", "germany/bund/acts"]


def test_exclude_wins(tree):
    sel = Selection(countries=["brazil", "united-states"],
                    exclude_corpora=["brazil/sao-paulo/acts"],
                    exclude_countries=["united-states"])
    assert [c.path for c in sel.resolve(tree)] == ["brazil/national/acts"]
    assert sel.is_selected(tree.by_path()["brazil/national/acts"])
    assert not sel.is_selected(tree.by_path()["brazil/sao-paulo/acts"])


def test_all_minus_exclusions(tree):
    sel = Selection(all=True, exclude_countries=["brazil"])
    paths = [c.path for c in sel.resolve(tree)]
    assert len(paths) == 5 and not any(p.startswith("brazil") for p in paths)


def test_unknown_names_raise(tree):
    with pytest.raises(KeyError):
        Selection(countries=["atlantis"]).resolve(tree)
    with pytest.raises(KeyError):
        Selection(corpora=["brazil/acre/acts"]).resolve(tree)


def test_include_and_exclude_edit_consistently():
    sel = Selection()
    sel.include(countries=["brazil"], corpora=["germany/bund/acts"])
    sel.exclude(corpora=["germany/bund/acts", "brazil/sao-paulo/acts"])
    assert sel.corpora == []
    assert sel.exclude_corpora == ["brazil/sao-paulo/acts", "germany/bund/acts"]
    sel.include(corpora=["germany/bund/acts"])
    assert sel.exclude_corpora == ["brazil/sao-paulo/acts"] and sel.corpora == ["germany/bund/acts"]
    assert not sel.is_empty()
    sel.clear()
    assert sel.is_empty()


def test_toml_round_trip(tmp_path: Path):
    sel = Selection(countries=["brazil", "germany"], corpora=["united-states/federal/usc"],
                    exclude_corpora=["brazil/sao-paulo/acts"], max_files_per_corpus=50)
    f = tmp_path / "selection.toml"
    sel.save(f)
    back = Selection.load(f)
    assert back == sel
    assert Selection.load(tmp_path / "missing.toml") == Selection()


def test_estimate_and_humanisation():
    stats = {
        "a": CorpusStats("a", None, 1000, 1_000_000),
        "b": CorpusStats("b", None, 100, 500_000),
    }
    e = estimate(stats, chunks_per_second=50)
    assert (e.corpora, e.files, e.bytes) == (2, 1100, 1_500_000)
    assert e.chunks == int(1100 * 1.05)
    assert e.seconds == pytest.approx(e.chunks / 50)
    capped = estimate(stats, 50, max_files_per_corpus=200)
    assert capped.files == 300 and capped.bytes == 200_000 + 500_000
    assert human_bytes(512) == "512 B" and human_bytes(1536) == "1.5 KB"
    assert human_duration(59) == "59s" and human_duration(3660) == "1h 01m"
    assert human_duration(3 * 86400) == "3d 00h"
