from __future__ import annotations

import os
from pathlib import Path

import pytest

from global_law_vector.cli import main

from .conftest import FIXTURE_TREE


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GLOBAL_LAW_PATH", str(FIXTURE_TREE))
    monkeypatch.setenv("GLV_SELECTION", str(tmp_path / "selection.toml"))
    monkeypatch.setenv("GLV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_select_saves_and_shows(env, capsys):
    assert main(["select", "--country", "brazil", "--corpus", "germany/bund/acts"]) == 0
    assert (env / "selection.toml").is_file()
    assert main(["select", "--show"]) == 0
    out = capsys.readouterr().out
    assert "brazil/sao-paulo/acts" in out and "germany/bund/acts" in out
    assert "3 corpora · 3 files" in out
    assert main(["select", "--exclude", "brazil/sao-paulo/acts", "--show", "--paths"]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-2:] == ["brazil/national/acts", "germany/bund/acts"]


def test_select_rejects_unknown(env, capsys):
    assert main(["select", "--country", "atlantis"]) == 2
    assert "unknown country" in capsys.readouterr().err
    assert not (env / "selection.toml").exists()


def test_empty_selection(env, capsys):
    assert main(["select", "--show"]) == 0
    assert "selection is empty" in capsys.readouterr().out


def test_list_marks_selection(env, capsys):
    main(["select", "--country", "malta"])
    assert main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    marked = [ln for ln in lines if ln.startswith("x")]
    assert len(marked) == 1 and "malta/national/acts" in marked[0]
    assert "7 corpora in 5 countries" in lines[-1]


def test_index_dry_run_counts(env, capsys):
    assert main(["index", "--dry-run", "--offline", "--all"]) == 0
    out = capsys.readouterr().out
    assert "7 corpora · 7 files · 7 chunks" in out
    assert "tokenizer: heuristic" in out


def test_index_dry_run_with_max_files(env, capsys):
    assert main(["index", "--dry-run", "--offline", "--country", "brazil", "--max-files", "1"]) == 0
    assert "2 corpora · 2 files" in capsys.readouterr().out


def test_missing_tree_is_an_error(env, monkeypatch, capsys):
    monkeypatch.setenv("GLOBAL_LAW_PATH", str(env / "nowhere"))
    assert main(["select", "--show"]) == 2
    assert "no global-law tree" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "glv 0.1.0" in capsys.readouterr().out


def test_dotenv_does_not_override_environment(env, monkeypatch):
    from global_law_vector.config import Config

    (env / ".env").write_text("GLV_CHUNK_TOKENS=77\nGLV_WORKERS=9\n")
    monkeypatch.setenv("GLV_WORKERS", "2")
    cfg = Config.from_env(env / ".env")
    assert cfg.chunk_tokens == 77 and cfg.workers == 2
    assert os.environ["GLV_CHUNK_TOKENS"] == "77"


def test_check_reports_unreachable_services(env, monkeypatch, capsys):
    monkeypatch.setenv("TEI_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:9/none")
    assert main(["check"]) == 1
    out = capsys.readouterr().out
    assert "7 corpora" in out and "UNREACHABLE" in out and "docker network connect" in out


def test_index_fails_fast_when_unreachable(env, monkeypatch, capsys):
    monkeypatch.setenv("TEI_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:9/none")
    assert main(["index", "--offline", "--country", "malta"]) == 1
    assert "not reachable" in capsys.readouterr().err
