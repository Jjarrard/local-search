import zipfile

from localsearch import extract


def test_blank_lines_split_and_hard_wraps_join():
    text = "First line\nwraps here.\n\nSecond para.\n\n\n  Third.  "
    assert extract.split_paragraphs(text) == ["First line wraps here.", "Second para.", "Third."]


def test_one_paragraph_per_long_line_is_split_per_line():
    a = "A" * 10 + " word" * 60 + "."
    b = "B" * 10 + " word" * 60 + "."
    c = "C" * 10 + " word" * 60 + "."
    paras = extract.split_paragraphs("\n".join([a, b, c]))
    assert len(paras) == 3


def test_long_paragraph_splits_at_sentences_under_limit():
    para = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = extract.split_long(para, limit=300)
    assert all(len(p) <= 300 for p in pieces)
    assert " ".join(pieces) == para


def test_sentence_without_full_stops_still_gets_cut():
    pieces = extract.split_long("word " * 1000, limit=200)
    assert all(len(p) <= 200 for p in pieces)


def test_html_blocks_become_paragraphs_and_scripts_vanish():
    html = "<html><head><title>T</title><style>p{}</style></head><body><h1>Head</h1><p>One &amp; two.</p><script>x()</script><p>Three.</p></body></html>"
    assert extract.split_paragraphs(extract.html_to_text(html)) == ["Head", "One & two.", "Three."]


def test_latin1_text_file_is_read(tmp_path):
    p = tmp_path / "old.txt"
    p.write_bytes("caf\xe9 society.".encode("latin-1"))
    assert extract.read_paragraphs(p) == ["café society."]


def _make_epub(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf"><manifest>'
            '<item id="c1" href="ch1.xhtml"/><item id="c2" href="ch2.xhtml"/></manifest>'
            '<spine><itemref idref="c2"/><itemref idref="c1"/></spine></package>',
        )
        zf.writestr("OEBPS/ch1.xhtml", "<html><body><p>Chapter one text.</p></body></html>")
        zf.writestr("OEBPS/ch2.xhtml", "<html><body><p>Chapter two text.</p></body></html>")


def test_epub_is_read_in_spine_order(tmp_path):
    p = tmp_path / "book.epub"
    _make_epub(p)
    # spine says ch2 first - reading order, not filename order
    assert extract.read_paragraphs(p) == ["Chapter two text.", "Chapter one text."]


def test_broken_epub_raises_extract_error(tmp_path):
    p = tmp_path / "bad.epub"
    p.write_text("not a zip")
    try:
        extract.read_paragraphs(p)
    except extract.ExtractError:
        return
    raise AssertionError("expected ExtractError")
