from __future__ import annotations

from global_law_vector.chunk import chunk_document, embedding_input
from global_law_vector.parse import Document, parse_document
from global_law_vector.tokenizer import HeuristicTokenizer

from .conftest import read


def _doc(body: str, title: str = "T") -> Document:
    return Document(path="c/l/x/f.md", corpus_path="c/l/x", title=title, body=body)


def test_small_document_is_one_chunk(tok):
    d = parse_document("germany/bund/acts/solzg/0001.md", "germany/bund/acts",
                       read("germany/bund/acts/solzg/0001.md"))
    [c] = chunk_document(d, tok, target_tokens=1024)
    assert c.ordinal == 0 and c.heading_path is None and c.text == d.body
    assert c.token_count == tok.count(d.body)


def test_empty_body_gives_nothing(tok):
    assert chunk_document(_doc("   \n"), tok) == []


def test_sections_keep_heading_path(tok):
    para = "word " * 60  # ~80 heuristic tokens
    body = (
        "# Act\n\nintro paragraph.\n\n## Chapter I\n\n" + para + "\n\n" + para +
        "\n\n### § 1\n\n" + para + "\n\n## Chapter II\n\n" + para
    )
    chunks = chunk_document(_doc(body), tok, target_tokens=100, max_tokens=1000)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    paths = [c.heading_path for c in chunks]
    assert paths[0] == "Act"
    assert "Act > Chapter I" in paths
    assert "Act > Chapter I > § 1" in paths
    assert paths[-1] == "Act > Chapter II"
    assert all(c.token_count <= 100 or c.token_count <= tok.count(para) for c in chunks)


def test_paragraphs_pack_up_to_target(tok):
    small = "alpha beta gamma."  # ~5 tokens
    body = "\n\n".join([small] * 20)
    chunks = chunk_document(_doc(body), tok, target_tokens=30, max_tokens=1000)
    assert len(chunks) > 1
    assert all(c.token_count <= 30 for c in chunks)
    assert "\n\n".join(c.text for c in chunks) == body


def test_oversize_paragraph_is_hard_split_under_max(tok):
    sentence = "This is a sentence about land registration. "
    body = sentence * 200  # one paragraph, ~2000 tokens
    chunks = chunk_document(_doc(body), tok, target_tokens=100, max_tokens=150)
    assert len(chunks) > 5
    assert all(c.token_count <= 150 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert joined.split() == body.split()


def test_unbreakable_run_is_split_on_words(tok):
    body = " ".join(f"w{i}" for i in range(500))  # no sentence ends at all
    chunks = chunk_document(_doc(body), tok, target_tokens=50, max_tokens=60)
    assert all(c.token_count <= 60 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == body.split()


def test_embedding_input_adds_title_and_heading(tok):
    d = _doc("x", title="Code civil")
    [c] = chunk_document(d, tok)
    assert embedding_input(d, c) == "Code civil\n\nx"
    c.heading_path = "Livre I > Art. 1"
    expected = "passage: Code civil > Livre I > Art. 1\n\nx"
    assert embedding_input(d, c, prefix="passage: ") == expected


def test_heuristic_tokenizer_monotone():
    t = HeuristicTokenizer()
    assert t.count("") == 0
    assert t.count("one two three") < t.count("one two three four five six")
    assert t.count("日本の法律") > t.count("law")
