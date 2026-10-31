"""Stage 6, EMBED: a client for Hugging Face text-embeddings-inference (TEI).

Standard library only (urllib); batches run on a thread pool, with retry and back-off on
connection errors, 429 and 5xx. Results keep the input order.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelInfo:
    model_id: str
    dims: int | None
    max_input_length: int | None
    raw: dict


class TEIClient:
    def __init__(self, base_url: str, timeout: float = 120.0, retries: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    # --- plumbing ---

    def _request(self, path: str, payload: dict | None = None) -> object:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        req.add_header("Content-Type", "application/json")
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode()
                    return json.loads(body) if body.strip() else None
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:300]
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    last = exc
                    log.warning("TEI %s -> %s, retrying in %.0fs: %s", path, exc.code, delay, body)
                else:
                    raise EmbeddingError(f"TEI {path} failed with HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
                if attempt >= self.retries:
                    break
                log.warning("TEI %s unreachable (%s), retrying in %.0fs", path, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise EmbeddingError(f"TEI {path} failed after {self.retries} retries: {last}")

    # --- API ---

    def health(self) -> bool:
        try:
            self._request("/health")
            return True
        except EmbeddingError:
            return False

    def info(self) -> ModelInfo:
        raw = self._request("/info")
        if not isinstance(raw, dict):
            raise EmbeddingError(f"unexpected /info payload: {raw!r}")
        dims = raw.get("hidden_size") or raw.get("dim")
        return ModelInfo(
            model_id=str(raw.get("model_id", "")),
            dims=int(dims) if dims else None,
            max_input_length=int(raw["max_input_length"]) if raw.get("max_input_length") else None,
            raw=raw,
        )

    def embed(self, texts: Sequence[str], truncate: bool = True) -> list[list[float]]:
        if not texts:
            return []
        out = self._request("/embed", {"inputs": list(texts), "truncate": truncate})
        if not isinstance(out, list) or len(out) != len(texts):
            raise EmbeddingError("TEI returned a malformed /embed response")
        return out

    def embed_many(
        self, texts: Sequence[str], batch_size: int = 32, workers: int = 4
    ) -> list[list[float]]:
        """Embed any number of texts in parallel batches, preserving order."""
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        if len(batches) <= 1 or workers <= 1:
            return [v for b in batches for v in self.embed(b)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self.embed, batches))
        return [v for b in results for v in b]


def check_model(info: ModelInfo, expected_model: str, expected_dims: int) -> None:
    """Refuse to proceed when the server's model is not the configured one (premise P5)."""
    if info.model_id and info.model_id != expected_model:
        raise EmbeddingError(
            f"embeddings server serves {info.model_id!r} but GLV_MODEL is {expected_model!r}"
        )
    if info.dims and info.dims != expected_dims:
        raise EmbeddingError(f"server embeds {info.dims} dims but GLV_DIMS is {expected_dims}")


def vector_literal(values: Iterable[float]) -> str:
    """pgvector's text form, accepted by both vector and halfvec columns."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"
