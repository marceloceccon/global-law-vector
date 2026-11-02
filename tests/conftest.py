from __future__ import annotations

import os
from pathlib import Path

import pytest

from global_law_vector.discover import Tree, discover
from global_law_vector.tokenizer import HeuristicTokenizer

FIXTURE_TREE = Path(__file__).parent / "fixtures" / "tree"


@pytest.fixture(autouse=True)
def _restore_environment():
    """Loading a .env writes into os.environ; undo it so no test leaks settings into the next."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(scope="session")
def tree() -> Tree:
    return discover(FIXTURE_TREE)


@pytest.fixture(scope="session")
def tok() -> HeuristicTokenizer:
    return HeuristicTokenizer()


def read(rel: str) -> bytes:
    return (FIXTURE_TREE / rel).read_bytes()
