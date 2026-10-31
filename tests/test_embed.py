"""TEI client against a stub HTTP server running in a thread."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from global_law_vector.embed import EmbeddingError, TEIClient, check_model, vector_literal


class _Stub(BaseHTTPRequestHandler):
    fail_next = 0
    calls: list[list[str]] = []

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/info":
            body = json.dumps({"model_id": "stub/model", "hidden_size": 3,
                               "max_input_length": 8192}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    seen_auth = None
    reject_auth = False

    def do_POST(self):
        _Stub.seen_auth = self.headers.get("Authorization")
        if _Stub.reject_auth:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"unauthorised")
            return
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        if _Stub.fail_next > 0:
            _Stub.fail_next -= 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"busy")
            return
        inputs = payload["inputs"]
        _Stub.calls.append(inputs)
        out = [[float(len(t)), 0.5, -1.0] for t in inputs]
        body = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_info_and_health(server):
    c = TEIClient(server)
    assert c.health()
    info = c.info()
    assert (info.model_id, info.dims, info.max_input_length) == ("stub/model", 3, 8192)
    check_model(info, "stub/model", 3)
    with pytest.raises(EmbeddingError):
        check_model(info, "other/model", 3)
    with pytest.raises(EmbeddingError):
        check_model(info, "stub/model", 768)
    info.dims = None  # /info without a width: measured through the client
    assert c.probe_dims() == 3
    check_model(info, "stub/model", 3, c)
    with pytest.raises(EmbeddingError):
        check_model(info, "stub/model", 4, c)


def test_embed_many_keeps_order_and_batches(server):
    c = TEIClient(server)
    _Stub.calls.clear()
    texts = [f"t{i}" * (i + 1) for i in range(10)]
    vectors = c.embed_many(texts, batch_size=3, workers=2)
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]
    assert sorted(len(b) for b in _Stub.calls) == [1, 3, 3, 3]
    assert c.embed([]) == []


def test_retry_on_503(server):
    c = TEIClient(server, retries=3)
    _Stub.fail_next = 2
    import global_law_vector.embed as m

    m.time.sleep = lambda s: None  # no waiting in tests
    assert c.embed(["x"])[0][0] == 1.0


def test_gives_up_after_retries(server):
    c = TEIClient(server, retries=1)
    _Stub.fail_next = 5
    import global_law_vector.embed as m

    m.time.sleep = lambda s: None
    with pytest.raises(EmbeddingError):
        c.embed(["x"])
    _Stub.fail_next = 0


def test_unreachable_server_raises_quickly():
    c = TEIClient("http://127.0.0.1:9", retries=0, timeout=1)
    assert not c.health()


def test_vector_literal():
    assert vector_literal([1, 0.5, -0.1234567891]) == "[1,0.5,-0.1234568]"


def test_api_key_is_sent_as_bearer(server):
    _Stub.seen_auth = None
    c = TEIClient(server, api_key="secret-key")
    c.embed(["x"])
    assert _Stub.seen_auth == "Bearer secret-key"
    _Stub.seen_auth = None
    TEIClient(server).embed(["x"])
    assert _Stub.seen_auth is None


def test_unauthorised_is_not_retried(server):
    _Stub.reject_auth = True
    try:
        with pytest.raises(EmbeddingError, match="TEI_API_KEY"):
            TEIClient(server, retries=3).embed(["x"])
    finally:
        _Stub.reject_auth = False
