"""Stage 1, DISCOVER: which corpora the global-law tree holds, pinned at which commit, with
how many provision files and bytes.

The root tree pins every corpus as a git submodule (525 on the 2026-10-29 build) while
`manifest.json` describes a subset of them (107) with language, coverage and record limits. The
submodule list is therefore the source of truth and the manifest is metadata.

Nothing here writes to the tree (premise P1). The only git command used is `git ls-tree`, read
from the root repository; when git or the repository is absent, `.gitmodules` is parsed instead
and the commit is unknown.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

SKIP_NAMES = frozenset(
    {"README.md", "REPRODUCE.md", "SOURCES.md", "CHANGELOG.md", "LICENSE.md", "MCP.md", "LICENSE"}
)
"""Markdown files that are documentation, not law."""


@dataclass(slots=True)
class Corpus:
    path: str  # "germany/bund/acts", relative to the root, forward slashes
    country: str  # "germany"
    level: str  # "bund"
    name: str  # "acts"
    commit_sha: str | None = None
    label: str | None = None  # from the manifest ("Germany")
    code: str | None = None  # ISO 3166-1 ("DE")
    language: str | None = None
    coverage: str | None = None
    record_to: str | None = None
    in_manifest: bool = False

    @classmethod
    def from_path(cls, path: str) -> Corpus:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            raise ValueError(f"corpus path must be country/level/corpus, got {path!r}")
        return cls(path="/".join(parts), country=parts[0], level=parts[1], name="/".join(parts[2:]))

    @property
    def short(self) -> str:
        """The corpus without its country: "bund/acts"."""
        return f"{self.level}/{self.name}"


@dataclass(slots=True)
class CorpusStats:
    path: str
    commit_sha: str | None
    file_count: int
    byte_count: int
    scanned_at: float = 0.0


@dataclass(slots=True)
class Tree:
    """The global-law tree: its root and the corpora it pins."""

    root: Path
    corpora: list[Corpus] = field(default_factory=list)

    def by_path(self) -> dict[str, Corpus]:
        return {c.path: c for c in self.corpora}

    def countries(self) -> dict[str, list[Corpus]]:
        out: dict[str, list[Corpus]] = {}
        for c in self.corpora:
            out.setdefault(c.country, []).append(c)
        return out


# --- pinned submodules -------------------------------------------------------------------------


def _gitlinks_from_git(root: Path) -> dict[str, str] | None:
    """{path: sha} for every gitlink in the root tree, or None if git cannot answer."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    links: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        # "<mode> SP <type> SP <sha> TAB <path>"
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "commit":
            links[path] = parts[2]
    return links


def _gitlinks_from_gitmodules(root: Path) -> dict[str, str | None]:
    """{path: None} for every submodule in .gitmodules (the commit is not knowable from it)."""
    links: dict[str, str | None] = {}
    gm = root / ".gitmodules"
    if not gm.is_file():
        return links
    for raw in gm.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("path") and "=" in line:
            links[line.split("=", 1)[1].strip()] = None
    return links


def _load_manifest(root: Path) -> dict[str, dict]:
    mf = root / "manifest.json"
    if not mf.is_file():
        return {}
    data = json.loads(mf.read_text(encoding="utf-8"))
    return {c["path"]: c for c in data.get("corpora", []) if "path" in c}


def discover(root: Path) -> Tree:
    """Enumerate the corpora pinned in the tree at `root`, with manifest metadata when present."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"global-law tree not found at {root}")
    links: dict[str, str | None] | None = _gitlinks_from_git(root)
    if not links:
        links = _gitlinks_from_gitmodules(root)
    manifest = _load_manifest(root)
    corpora: list[Corpus] = []
    for path in sorted(set(links) | set(manifest)):
        if not (root / path).is_dir():
            continue
        c = Corpus.from_path(path)
        c.commit_sha = links.get(path)
        m = manifest.get(path)
        if m:
            c.in_manifest = True
            c.label = m.get("label") or m.get("name")
            c.code = m.get("code") or None
            c.language = m.get("language") or None
            c.coverage = m.get("coverage") or None
            c.record_to = m.get("record_to") or None
        corpora.append(c)
    return Tree(root=root, corpora=corpora)


# --- files -------------------------------------------------------------------------------------


def is_law_file(name: str) -> bool:
    return name.endswith(".md") and name not in SKIP_NAMES


def iter_files(root: Path, corpus: Corpus) -> Iterator[Path]:
    """Every provision file of a corpus, relative to the root, in a stable order."""
    base = root / corpus.path
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if is_law_file(name):
                yield Path(dirpath, name).relative_to(root)


def scan(root: Path, corpus: Corpus) -> CorpusStats:
    """Count files and bytes of a corpus (fast: a stat per file, no reads)."""
    import time

    files = 0
    size = 0
    for rel in iter_files(root, corpus):
        files += 1
        size += (root / rel).stat().st_size
    return CorpusStats(corpus.path, corpus.commit_sha, files, size, time.time())


class StatsCache:
    """Per-corpus stats keyed by (path, commit), kept as one JSON file under the cache dir.

    The first scan of the whole tree touches 4.5 million inodes; later runs read this file.
    """

    def __init__(self, cache_dir: Path, root: Path) -> None:
        tag = hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:12]
        self.file = Path(cache_dir) / f"stats-{tag}.json"
        self._data: dict[str, dict] = {}
        if self.file.is_file():
            try:
                self._data = json.loads(self.file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def _key(corpus: Corpus) -> str:
        return f"{corpus.path}@{corpus.commit_sha or 'unknown'}"

    def get(self, corpus: Corpus) -> CorpusStats | None:
        d = self._data.get(self._key(corpus))
        return CorpusStats(**d) if d else None

    def put(self, stats: CorpusStats) -> None:
        self._data[f"{stats.path}@{stats.commit_sha or 'unknown'}"] = asdict(stats)

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=0), encoding="utf-8")
        tmp.replace(self.file)


def stats_for(
    tree: Tree, corpora: list[Corpus], cache: StatsCache | None = None, progress=None
) -> dict[str, CorpusStats]:
    """Stats for the given corpora, scanning only those missing from the cache."""
    out: dict[str, CorpusStats] = {}
    dirty = False
    for i, c in enumerate(corpora):
        s = cache.get(c) if cache else None
        if s is None:
            if progress:
                progress(c, i, len(corpora))
            s = scan(tree.root, c)
            if cache:
                cache.put(s)
                dirty = True
        out[c.path] = s
    if cache and dirty:
        cache.save()
    return out
