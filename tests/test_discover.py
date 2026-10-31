from __future__ import annotations

from pathlib import Path

from global_law_vector.discover import (
    Corpus,
    StatsCache,
    is_law_file,
    iter_files,
    scan,
    stats_for,
)

from .conftest import FIXTURE_TREE


def test_corpora_come_from_gitmodules_with_manifest_metadata(tree):
    paths = [c.path for c in tree.corpora]
    # nauru is in .gitmodules but has no directory in the fixture: not listed
    assert "nauru/national/acts" not in paths
    assert paths == sorted(paths)
    assert len(paths) == 7
    by = tree.by_path()
    de = by["germany/bund/acts"]
    assert (de.country, de.level, de.name, de.short) == ("germany", "bund", "acts", "bund/acts")
    assert de.in_manifest and de.code == "DE" and de.language == "de"
    sp = by["brazil/sao-paulo/acts"]
    assert not sp.in_manifest and sp.code is None
    # the fixture tree is not a git repository, so the commit is unknown
    assert de.commit_sha is None


def test_countries_group_corpora(tree):
    countries = tree.countries()
    assert [c.short for c in countries["brazil"]] == ["national/acts", "sao-paulo/acts"]
    assert len(countries["united-states"]) == 2


def test_corpus_from_path_requires_three_segments():
    import pytest

    with pytest.raises(ValueError):
        Corpus.from_path("germany/bund")
    c = Corpus.from_path("/a/b/c/d/")
    assert (c.country, c.level, c.name) == ("a", "b", "c/d")


def test_skip_rules():
    assert is_law_file("0001.md")
    assert not is_law_file("README.md")
    assert not is_law_file("SOURCES.md")
    assert not is_law_file("notes.txt")


def test_iter_files_skips_documentation(tree):
    de = tree.by_path()["germany/bund/acts"]
    files = [f.as_posix() for f in iter_files(tree.root, de)]
    assert files == ["germany/bund/acts/solzg/0001.md"]


def test_scan_counts_bytes(tree):
    de = tree.by_path()["germany/bund/acts"]
    s = scan(tree.root, de)
    assert s.file_count == 1
    assert s.byte_count == (FIXTURE_TREE / "germany/bund/acts/solzg/0001.md").stat().st_size


def test_stats_cache_round_trip(tree, tmp_path: Path):
    cache = StatsCache(tmp_path, tree.root)
    corpora = tree.corpora[:2]
    calls = []
    first = stats_for(tree, corpora, cache, progress=lambda c, i, n: calls.append(c.path))
    assert len(calls) == 2 and cache.file.is_file()
    cache2 = StatsCache(tmp_path, tree.root)
    calls.clear()
    second = stats_for(tree, corpora, cache2, progress=lambda c, i, n: calls.append(c.path))
    assert calls == []
    assert {k: (v.file_count, v.byte_count) for k, v in first.items()} == {
        k: (v.file_count, v.byte_count) for k, v in second.items()
    }
