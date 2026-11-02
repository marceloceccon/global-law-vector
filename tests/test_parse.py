from __future__ import annotations

from global_law_vector.parse import first_heading, parse_document, parse_front_matter

from .conftest import read


def _doc(rel: str, corpus: str):
    return parse_document(rel, corpus, read(rel))


def test_germany_heading_and_body():
    d = _doc("germany/bund/acts/solzg/0001.md", "germany/bund/acts")
    assert d.title == "§ 1 Erhebung eines Solidaritätszuschlags"
    assert d.body.startswith("# § 1")
    assert "Ergänzungsabgabe" in d.body
    assert d.front_matter == {} and d.citation is None
    assert len(d.content_sha) == 64 and d.byte_size == 156
    assert d.relative_in_corpus == "solzg/0001.md"


def test_czechia_front_matter():
    d = _doc("czechia/national/acts/sb/1919/213.md", "czechia/national/acts")
    assert d.front_matter["citace"] == "213/1919 Sb."
    assert d.front_matter["druh"] == "Zákon"
    assert d.front_matter["eli"].startswith("https://")
    assert d.citation == "213/1919 Sb."
    assert d.title == "Zákon o sloučení sousedních obcí s Brnem."
    assert not d.body.startswith("---")
    assert "### § 1." in d.body


def test_sao_paulo_bare_text_titles_from_path():
    d = _doc("brazil/sao-paulo/acts/lei-1990-07004/art-0001.md", "brazil/sao-paulo/acts")
    assert d.title == "lei-1990-07004 / art-0001"
    assert d.body.startswith("Artigo 1.º")


def test_brazil_national_bare_text():
    d = _doc("brazil/national/acts/codigo-civil/art-0700.md", "brazil/national/acts")
    assert d.title == "codigo-civil / art-0700"
    assert d.body.startswith("Art. 700.")


def test_malta_prefers_h1_over_earlier_h2():
    d = _doc("malta/national/acts/cap/296/48.md", "malta/national/acts")
    assert d.title == "48. Conversion of title into a guaranteed title."


def test_california_and_usc():
    ca = _doc("united-states/california/statutes/ins/1182.md", "united-states/california/statutes")
    assert ca.title == "California Insurance Code § 1182"
    assert "Source: Amended by" in ca.body
    us = _doc("united-states/federal/usc/title-11/1182.md", "united-states/federal/usc")
    assert us.title == "11 U.S.C. § 1182. Definitions"


def test_front_matter_scalars():
    fm, rest = parse_front_matter(
        '---\nname: "quoted"\nn: 12\nflag: true\nnothing: null\n'
        'list: [a, "b"]\nplain: x y\n---\nbody'
    )
    assert fm == {
        "name": "quoted",
        "n": 12,
        "flag": True,
        "nothing": None,
        "list": ["a", "b"],
        "plain": "x y",
    }
    assert rest == "body"


def test_front_matter_absent_or_unterminated():
    assert parse_front_matter("# h\n---\nx") == ({}, "# h\n---\nx")
    assert parse_front_matter("---\na: 1\nno end") == ({}, "---\na: 1\nno end")


def test_first_heading_fallbacks():
    assert first_heading("text\n## two\n### three") == "two"
    assert first_heading("no headings") is None
    assert first_heading("# trailing hashes ##") == "trailing hashes"


def test_bom_and_title_from_front_matter():
    d = parse_document("x/y/z/a.md", "x/y/z", "﻿---\ntitle: From FM\n---\nbody only".encode())
    assert d.title == "From FM" and d.body == "body only"


def test_front_matter_block_syntax():
    from global_law_vector.parse import parse_front_matter

    text = (
        "---\n"
        "nazev: Zákon o obcích\n"
        "tags:\n"
        "  - obce\n"
        "  - samospráva\n"
        "extent:\n"
        "note: |\n"
        "  First line.\n"
        "  Second line.\n"
        "summary: >\n"
        "  Folded\n"
        "  into one.\n"
        "source:\n"
        "  url: https://example.org\n"
        "  retrieved: 2026-10-29\n"
        "---\n"
        "# § 1\n"
    )
    fm, rest = parse_front_matter(text)
    assert fm["nazev"] == "Zákon o obcích"
    assert fm["tags"] == ["obce", "samospráva"]
    assert fm["extent"] == ""
    assert fm["note"] == "First line.\nSecond line."
    assert fm["summary"] == "Folded into one."
    assert fm["source"] == "url: https://example.org\nretrieved: 2026-10-29"
    assert rest == "# § 1\n"


def test_invalid_utf8_is_replaced_and_logged(caplog):
    from global_law_vector.parse import parse_document

    raw = "# § 1\n\nPlatí od 1. ledna.".encode() + b"\xff\xfe"
    with caplog.at_level("WARNING"):
        doc = parse_document("c/l/x/f.md", "c/l/x", raw)
    assert "�" in doc.body
    assert "not valid UTF-8" in caplog.text and "c/l/x/f.md" in caplog.text
