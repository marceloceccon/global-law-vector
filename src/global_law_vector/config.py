"""Configuration from environment variables and an optional .env file.

Every value that depends on the embedding model (its id, dimensions, input limit, query prefix)
lives here and is recorded in the database by the store, so a change is detected (premise P5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-m-v2.0"
DEFAULT_DIMS = 768
DEFAULT_MAX_TOKENS = 8192
DEFAULT_QUERY_PREFIX = "query: "


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read KEY=VALUE lines from a .env file without overriding the real environment."""
    path = path or Path(".env")
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            values[key] = value
    return values


def _find_global_law() -> Path:
    """Locate the global-law tree: the env var, then ./global-law, then ../global-law."""
    env = os.environ.get("GLOBAL_LAW_PATH")
    if env:
        return Path(env).expanduser()
    for candidate in (Path("global-law"), Path("../global-law"), Path("../../global-law")):
        if (candidate / "manifest.json").is_file():
            return candidate.resolve()
    return Path("global-law")


@dataclass(slots=True)
class Config:
    global_law: Path = field(default_factory=_find_global_law)
    database_url: str = "postgresql://utility:utility@127.0.0.1:5432/utility"
    schema: str = "global_law"
    tei_url: str = "http://127.0.0.1:8080"
    model: str = DEFAULT_MODEL
    dims: int = DEFAULT_DIMS
    max_tokens: int = DEFAULT_MAX_TOKENS
    query_prefix: str = DEFAULT_QUERY_PREFIX
    document_prefix: str = ""
    vector_type: str = "halfvec"  # or "vector"
    chunk_tokens: int = 1024
    chunk_overlap: int = 0
    batch_size: int = 16  # TEI's default --max-client-batch-size is 32
    workers: int = 2
    tei_timeout: float = 600.0  # a CPU server can take minutes per batch
    tokenizer_path: Path | None = None
    selection_file: Path = Path("selection.toml")
    cache_dir: Path = Path.home() / ".cache" / "global-law-vector"
    chunks_per_second: float = 1.5  # measured on a 6-core CPU; replaced by the stored figure

    @classmethod
    def from_env(cls, dotenv: Path | None = None) -> Config:
        load_dotenv(dotenv)
        env = os.environ
        cfg = cls()
        cfg.database_url = env.get("DATABASE_URL", cfg.database_url)
        cfg.schema = env.get("GLV_SCHEMA", cfg.schema)
        cfg.tei_url = env.get("TEI_URL", cfg.tei_url).rstrip("/")
        cfg.model = env.get("GLV_MODEL", cfg.model)
        cfg.dims = int(env.get("GLV_DIMS", cfg.dims))
        cfg.max_tokens = int(env.get("GLV_MAX_TOKENS", cfg.max_tokens))
        cfg.query_prefix = env.get("GLV_QUERY_PREFIX", cfg.query_prefix)
        cfg.document_prefix = env.get("GLV_DOCUMENT_PREFIX", cfg.document_prefix)
        cfg.vector_type = env.get("GLV_VECTOR_TYPE", cfg.vector_type)
        cfg.chunk_tokens = int(env.get("GLV_CHUNK_TOKENS", cfg.chunk_tokens))
        cfg.chunk_overlap = int(env.get("GLV_CHUNK_OVERLAP", cfg.chunk_overlap))
        cfg.batch_size = int(env.get("GLV_BATCH_SIZE", cfg.batch_size))
        cfg.workers = int(env.get("GLV_WORKERS", cfg.workers))
        cfg.tei_timeout = float(env.get("GLV_TEI_TIMEOUT", cfg.tei_timeout))
        if env.get("GLV_TOKENIZER"):
            cfg.tokenizer_path = Path(env["GLV_TOKENIZER"]).expanduser()
        cfg.selection_file = Path(env.get("GLV_SELECTION", str(cfg.selection_file)))
        if env.get("GLV_CACHE_DIR"):
            cfg.cache_dir = Path(env["GLV_CACHE_DIR"]).expanduser()
        cfg.chunks_per_second = float(env.get("GLV_CHUNKS_PER_SECOND", cfg.chunks_per_second))
        if cfg.vector_type not in ("vector", "halfvec"):
            raise ValueError(
                f"GLV_VECTOR_TYPE must be 'vector' or 'halfvec', not {cfg.vector_type!r}"
            )
        return cfg
