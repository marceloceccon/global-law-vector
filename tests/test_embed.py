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

    def do_POST(self):
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
