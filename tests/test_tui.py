"""Headless smoke test of the Textual UI on the fixture tree (no database, no embeddings)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from .conftest import FIXTURE_TREE

textual = pytest.importorskip("textual")


def test_select_screen_toggles_and_saves(tmp_path: Path, monkeypatch):
    from textual.widgets import Static, Tree

    from global_law_vector.config import Config
    from global_law_vector.selection import Selection
    from global_law_vector.tui.app import GlvApp, SelectScreen

    monkeypatch.setenv("GLOBAL_LAW_PATH", str(FIXTURE_TREE))
    monkeypatch.setenv("GLV_SELECTION", str(tmp_path / "selection.toml"))
    monkeypatch.setenv("GLV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    cfg = Config.from_env()

    async def scenario() -> None:
        app = GlvApp(cfg)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SelectScreen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            tree = screen.query_one("#tree", Tree)
            assert len(tree.root.children) == 5  # countries in the fixture tree
            assert len(screen.state.stats) == 7
            # cursor on the first country (brazil): toggle selects both of its corpora
            tree.cursor_line = 0
            await pilot.press("space")
            await pilot.pause()
            assert screen.state.selection.countries == ["brazil"]
            label = str(tree.root.children[0].label)
            assert label.startswith("[x] brazil")
            totals = str(screen.query_one("#totals", Static).content)
            assert "Selected: 2 corpora · 2 files" in totals
            # toggle again deselects
            await pilot.press("space")
            await pilot.pause()
            assert screen.state.selection.is_empty()
            # filter narrows the tree
            await pilot.press("slash")
            await pilot.press(*"malta")
            await pilot.pause()
            assert [n.data[1] for n in tree.root.children] == ["malta"]
            await pilot.press("escape")
            await pilot.press("a", "s")
            await pilot.pause()
            assert Selection.load(cfg.selection_file).all is True

    asyncio.run(scenario())
