"""Stage 2, SELECT: which corpora to index, and what that choice costs.

A selection is countries plus individual corpora to include, minus corpora or countries to
exclude. The file form is `selection.toml`; the CLI and the TUI read and write the same object,
so an interactive choice is reproducible (premise P3).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .discover import Corpus, CorpusStats, Tree

# A first-run guess of how many chunks one provision file yields (most are one chunk).
CHUNKS_PER_FILE = 1.05


@dataclass(slots=True)
class Selection:
    countries: list[str] = field(default_factory=list)
    corpora: list[str] = field(default_factory=list)
    exclude_countries: list[str] = field(default_factory=list)
    exclude_corpora: list[str] = field(default_factory=list)
    all: bool = False
    max_files_per_corpus: int = 0

    # --- file form ---

    @classmethod
    def load(cls, path: Path) -> Selection:
        if not Path(path).is_file():
            return cls()
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        inc = data.get("include", {})
        exc = data.get("exclude", {})
        lim = data.get("limits", {})
        return cls(
            countries=sorted(set(inc.get("countries", []))),
            corpora=sorted(set(inc.get("corpora", []))),
            exclude_countries=sorted(set(exc.get("countries", []))),
            exclude_corpora=sorted(set(exc.get("corpora", []))),
            all=bool(inc.get("all", False)),
            max_files_per_corpus=int(lim.get("max_files_per_corpus", 0)),
        )

    def dump(self) -> str:
        def arr(items: list[str]) -> str:
            return "[" + ", ".join(_toml_str(i) for i in items) + "]"

        lines = [
            "# global-law-vector selection. Edit by hand or with `glv select` / `glv tui`.",
            "[include]",
            f"all = {'true' if self.all else 'false'}",
            f"countries = {arr(self.countries)}",
            f"corpora = {arr(self.corpora)}",
            "",
            "[exclude]",
            f"countries = {arr(self.exclude_countries)}",
            f"corpora = {arr(self.exclude_corpora)}",
            "",
            "[limits]",
            f"max_files_per_corpus = {self.max_files_per_corpus}",
            "",
        ]
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.dump(), encoding="utf-8")

    # --- editing ---

    def include(self, *, countries: list[str] = (), corpora: list[str] = ()) -> None:
        self.countries = sorted(set(self.countries) | set(countries))
        self.corpora = sorted(set(self.corpora) | set(corpora))
        self.exclude_countries = sorted(set(self.exclude_countries) - set(countries))
        self.exclude_corpora = sorted(set(self.exclude_corpora) - set(corpora))

    def exclude(self, *, countries: list[str] = (), corpora: list[str] = ()) -> None:
        self.exclude_countries = sorted(set(self.exclude_countries) | set(countries))
        self.exclude_corpora = sorted(set(self.exclude_corpora) | set(corpora))
        self.countries = sorted(set(self.countries) - set(countries))
        self.corpora = sorted(set(self.corpora) - set(corpora))

    def clear(self) -> None:
        self.countries, self.corpora = [], []
        self.exclude_countries, self.exclude_corpora = [], []
        self.all = False

    def is_empty(self) -> bool:
        return not (self.all or self.countries or self.corpora)

    # --- resolution ---

    def resolve(self, tree: Tree) -> list[Corpus]:
        """The corpora this selection names, in tree order. Unknown names raise."""
        known = tree.by_path()
        countries = tree.countries()
        for name in (*self.countries, *self.exclude_countries):
            if name not in countries:
                raise KeyError(f"unknown country {name!r}")
        for path in (*self.corpora, *self.exclude_corpora):
            if path not in known:
                raise KeyError(f"unknown corpus {path!r}")
        chosen: set[str] = set()
        if self.all:
            chosen = set(known)
        for name in self.countries:
            chosen.update(c.path for c in countries[name])
        chosen.update(self.corpora)
        for name in self.exclude_countries:
            chosen.difference_update(c.path for c in countries[name])
        chosen.difference_update(self.exclude_corpora)
        return [c for c in tree.corpora if c.path in chosen]

    def is_selected(self, corpus: Corpus) -> bool:
        if corpus.path in self.exclude_corpora or corpus.country in self.exclude_countries:
            return False
        return self.all or corpus.country in self.countries or corpus.path in self.corpora


@dataclass(slots=True)
class Estimate:
    corpora: int
    files: int
    bytes: int
    chunks: int
    seconds: float

    @property
    def human_time(self) -> str:
        return human_duration(self.seconds)

    @property
    def human_bytes(self) -> str:
        return human_bytes(self.bytes)


def estimate(
    stats: dict[str, CorpusStats],
    chunks_per_second: float,
    max_files_per_corpus: int = 0,
) -> Estimate:
    files = 0
    size = 0
    for s in stats.values():
        n = s.file_count
        b = s.byte_count
        if max_files_per_corpus and n > max_files_per_corpus:
            b = int(b * max_files_per_corpus / n)
            n = max_files_per_corpus
        files += n
        size += b
    chunks = int(files * CHUNKS_PER_FILE)
    seconds = chunks / chunks_per_second if chunks_per_second > 0 else 0.0
    return Estimate(len(stats), files, size, chunks, seconds)


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def human_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    h, rem = divmod(s, 3600)
    m = rem // 60
    if h >= 48:
        d, h = divmod(h, 24)
        return f"{d}d {h:02d}h"
    return f"{h}h {m:02d}m"


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
