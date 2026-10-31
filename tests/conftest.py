from __future__ import annotations

from pathlib import Path

import pytest

from global_law_vector.discover import Tree, discover
from global_law_vector.tokenizer import HeuristicTokenizer

FIXTURE_TREE = Path(__file__).parent / "fixtures" / "tree"


@pytest.fixture(scope="session")
def tree() -> Tree:
    return discover(FIXTURE_TREE)


@pytest.fixture(scope="session")
def tok() -> HeuristicTokenizer:
    return HeuristicTokenizer()


def read(rel: str) -> bytes:
    return (FIXTURE_TREE / rel).read_bytes()
