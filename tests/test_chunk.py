from __future__ import annotations

from global_law_vector.chunk import chunk_document, embedding_input
from global_law_vector.parse import Document, parse_document
from global_law_vector.tokenizer import HeuristicTokenizer

from .conftest import read


def _doc(body: str, title: str = "T") -> Document:
    return Document(path="c/l/x/f.md", corpus_path="c/l/x", title=title, body=body)


def test_small_document_is_one_chunk(tok):
    d = parse_document(
        "germany/bund/acts/solzg/0001.md",
        "germany/bund/acts",
        read("germany/bund/acts/solzg/0001.md"),
    )
    [c] = chunk_document(d, tok, target_tokens=1024)
    assert c.ordinal == 0 and c.heading_path is None and c.text == d.body
    assert c.token_count == tok.count(d.body)


def test_empty_body_gives_nothing(tok):
    assert chunk_document(_doc("   \n"), tok) == []


def test_sections_keep_heading_path(tok):
    para = "word " * 60  # ~80 heuristic tokens
    body = (
        "# Act\n\nintro paragraph.\n\n## Chapter I\n\n"
        + para
        + "\n\n"
        + para
        + "\n\n### § 1\n\n"
        + para
        + "\n\n## Chapter II\n\n"
        + para
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


def test_tokenizer_cache_round_trip(tmp_path, monkeypatch):
    """A tokenizer fetched once is saved under the cache dir and loaded from there next time."""
    pytest = __import__("pytest")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    from global_law_vector import tokenizer as mod
    from global_law_vector.tokenizer import HFTokenizer, cached_tokenizer_path, load_tokenizer

    fake = tokenizers.Tokenizer(models.WordLevel({"a": 0, "b": 1, "[UNK]": 2}, unk_token="[UNK]"))
    fake.pre_tokenizer = pre_tokenizers.Whitespace()
    calls = []

    def from_hub(cls, model):
        calls.append(model)
        return cls(fake, model)

    monkeypatch.setattr(HFTokenizer, "from_hub", classmethod(from_hub))
    t1 = load_tokenizer("org/model", cache_dir=tmp_path)
    assert calls == ["org/model"] and t1.count("a b a") == 3
    assert cached_tokenizer_path(tmp_path, "org/model").is_file()
    t2 = load_tokenizer("org/model", cache_dir=tmp_path)
    assert calls == ["org/model"]  # no second fetch
    assert t2.name == "org/model" and t2.count("a b a") == 3  # same identity as the hub copy
    assert mod.HeuristicTokenizer().name == "heuristic"


def test_heuristic_counts_cjk_and_hangul_per_character(tok):
    latin = tok.count("a b c d")
    assert tok.count("民法典第一条") >= 6
    assert tok.count("대한민국헌법") >= 6
    assert tok.count("大韓民国 憲法") > latin


def _word_level(vocab_words: list[str]):
    """A tiny real `tokenizers` tokenizer: one token per whitespace-separated word."""
    import pytest

    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    vocab = {w: i for i, w in enumerate([*vocab_words, "[UNK]", "[PAD]"])}
    t = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return t


def test_tokenizer_file_truncation_does_not_cap_counts(tmp_path):
    """Model tokenizer.json files ship with truncation (512 for arctic-embed-m-v2.0) and
    padding. Counting must ignore both, or no long document is ever split."""
    from global_law_vector.tokenizer import HFTokenizer

    raw = _word_level(["lei"])
    raw.enable_truncation(max_length=8)
    raw.enable_padding(length=16, pad_token="[PAD]")
    path = tmp_path / "tokenizer.json"
    raw.save(str(path))
    assert '"max_length": 8' in path.read_text()  # the file itself still says 8

    tok = HFTokenizer.from_file(path)
    assert tok.count("lei " * 50) == 50 and tok.count("lei") == 1
    chunks = chunk_document(_doc("lei " * 50), tok, target_tokens=20)
    assert len(chunks) == 3 and all(c.token_count <= 20 for c in chunks)


def test_paragraph_above_target_is_split_even_below_model_limit(tok):
    para = "This is a sentence of the act. " * 120  # ~1,000 heuristic tokens, one paragraph
    chunks = chunk_document(_doc(para), tok, target_tokens=200, max_tokens=8192)
    assert len(chunks) >= 5
    assert all(c.token_count <= 200 for c in chunks)


def test_splitting_keeps_line_breaks(tok):
    items = "\n".join(
        f"({i}) the holder shall notify the authority within {i} days;" for i in range(1, 60)
    )
    chunks = chunk_document(_doc(items), tok, target_tokens=120)
    assert len(chunks) > 1
    assert all("\n" in c.text for c in chunks)  # list items stay on their own lines
    assert all(c.token_count <= 120 for c in chunks)
    rejoined = "\n".join(c.text for c in chunks)
    assert rejoined.split() == items.split()  # nothing lost or reordered


def test_text_without_spaces_is_split_by_characters(tok):
    text = "民法典規定" * 400  # 2,000 CJK characters, no spaces, no punctuation
    chunks = chunk_document(_doc(text), tok, target_tokens=300)
    assert len(chunks) >= 7 and all(c.token_count <= 300 for c in chunks)
    assert "".join(c.text for c in chunks) == text


def test_long_unbroken_paragraph_needs_few_token_counts():
    """Splitting is O(n log n) in token counts, not one count per word."""
    calls = 0
    base = HeuristicTokenizer()

    class Counting:
        name, exact = "counting", False

        def count(self, text: str) -> int:
            nonlocal calls
            calls += 1
            return base.count(text)

    words = 20_000
    chunks = chunk_document(_doc("word " * words), Counting(), target_tokens=1024)
    assert len(chunks) > 20
    assert calls < words / 10


def test_title_is_not_sent_twice():
    title = "5. Renewals of tenancy and new tenancies."
    doc = _doc(f"# {title}\n\n(1) When a landlord intends to renew the letting...", title=title)
    [c] = chunk_document(doc, HeuristicTokenizer())
    sent = embedding_input(doc, c)
    assert sent.count(title) == 1 and sent.startswith("# 5. Renewals")
