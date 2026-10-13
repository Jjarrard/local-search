"""Every format reader, against small hand-built files.

The Office/ODF fixtures are the minimum XML each reader needs; the readers
were also checked against files written by Word-compatible tools
(python-docx, openpyxl, python-pptx, odfpy, macOS textutil).
"""

import zipfile

from localsearch import extract
from localsearch.pipeline import looks_like_junk

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ODF = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"'
)


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


def rels(*entries):
    body = "".join(f'<Relationship Id="{i}" Type="{REL}/{t}" Target="{target}"/>' for i, t, target in entries)
    return f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{body}</Relationships>'


def paras(path):
    return extract.read_paragraphs(path)


# --- office open xml -------------------------------------------------------


def test_docx_paragraphs_and_table(tmp_path):
    doc = (
        f'<w:document {W}><w:body>'
        '<w:p><w:r><w:t>Tortoise </w:t></w:r><w:r><w:t>Report</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>Eats</w:t><w:tab/><w:t>cactus.</w:t></w:r></w:p>'
        '<w:tbl>'
        '<w:tr><w:tc><w:p><w:r><w:t>Island</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Shell</w:t></w:r></w:p></w:tc></w:tr>'
        '<w:tr><w:tc><w:p><w:r><w:t>Hood</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Saddle</w:t></w:r></w:p></w:tc></w:tr>'
        '</w:tbl>'
        '<w:sdt><w:sdtContent><w:p><w:r><w:t>Inside a content control.</w:t></w:r></w:p></w:sdtContent></w:sdt>'
        '</w:body></w:document>'
    )
    path = make_zip(tmp_path / "a.docx", {"word/document.xml": doc})
    assert paras(path) == [
        "Tortoise Report",
        "Eats cactus.",
        "Island: Hood; Shell: Saddle.",
        "Inside a content control.",
    ]


def test_xlsx_sheets_shared_strings_and_header_rows(tmp_path):
    files = {
        "xl/workbook.xml": f'<workbook {S} {R}><sheets>'
        '<sheet name="Sales" sheetId="1" r:id="rId1"/><sheet name="Notes" sheetId="2" r:id="rId2"/>'
        '</sheets></workbook>',
        "xl/_rels/workbook.xml.rels": rels(("rId1", "worksheet", "worksheets/sheet1.xml"), ("rId2", "worksheet", "worksheets/sheet2.xml")),
        "xl/sharedStrings.xml": f'<sst {S}><si><t>Month</t></si><si><t>Revenue</t></si><si><t>Jan</t></si>'
        '<si><r><t>Rich </t></r><r><t>text note</t></r><rPh><t>ignored</t></rPh></si></sst>',
        "xl/worksheets/sheet1.xml": f'<worksheet {S}><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>1200</v></c></row>'
        '<row r="3"><c r="B3" t="inlineStr"><is><t>gap before me</t></is></c></row>'
        '</sheetData></worksheet>',
        "xl/worksheets/sheet2.xml": f'<worksheet {S}><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>3</v></c></row><row r="2"><c r="C2" t="b"><v>1</v></c></row>'
        '</sheetData></worksheet>',
    }
    path = make_zip(tmp_path / "a.xlsx", files)
    assert paras(path) == [
        "[Sales] Month: Jan; Revenue: 1200. Revenue: gap before me.",
        "[Notes] Rich text note. TRUE.",  # a lone cell is not a header row
    ]


def test_pptx_slides_in_presentation_order_with_notes(tmp_path):
    def slide(*texts):
        ps = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in texts)
        return f'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" {A}><a:txBody>{ps}</a:txBody></p:sld>'

    files = {
        # presentation order is slide2 then slide1 - file names don't decide
        "ppt/presentation.xml": f'<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" {R}>'
        '<p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId1"/></p:sldIdLst></p:presentation>',
        "ppt/_rels/presentation.xml.rels": rels(("rId1", "slide", "slides/slide1.xml"), ("rId2", "slide", "slides/slide2.xml")),
        "ppt/slides/slide1.xml": slide("Second", "Shells differ"),
        "ppt/slides/slide2.xml": slide("Galapagos Trip", "Tortoises eat cactus."),
        "ppt/slides/_rels/slide2.xml.rels": rels(("rId9", "notesSlide", "../notesSlides/notesSlide1.xml")),
        "ppt/notesSlides/notesSlide1.xml": slide("Mention the finches", "1"),
    }
    path = make_zip(tmp_path / "a.pptx", files)
    assert paras(path) == [
        "Slide 1: Galapagos Trip. Tortoises eat cactus. Notes: Mention the finches.",
        "Slide 2: Second. Shells differ.",
    ]


# --- opendocument ----------------------------------------------------------


def test_odt_headings_paragraphs_and_spacing_elements(tmp_path):
    content = (
        f'<office:document-content {ODF}><office:body><office:text>'
        '<text:h>Report</text:h><text:p>Two<text:s text:c="2"/>spaces<text:tab/>and<text:line-break/>breaks.</text:p>'
        '</office:text></office:body></office:document-content>'
    )
    path = make_zip(tmp_path / "a.odt", {"content.xml": content})
    assert paras(path) == ["Report", "Two spaces and breaks."]


def test_ods_repeated_cells_and_rows_dont_explode(tmp_path):
    content = (
        f'<office:document-content {ODF}><office:body><office:spreadsheet><table:table table:name="Animals">'
        '<table:table-row><table:table-cell><text:p>Animal</text:p></table:table-cell>'
        '<table:table-cell><text:p>Diet</text:p></table:table-cell><table:table-cell table:number-columns-repeated="16000"/></table:table-row>'
        '<table:table-row><table:table-cell><text:p>Tortoise</text:p></table:table-cell>'
        '<table:table-cell><text:p>Cactus</text:p></table:table-cell></table:table-row>'
        '<table:table-row table:number-rows-repeated="1048000"><table:table-cell table:number-columns-repeated="16000"/></table:table-row>'
        '</table:table></office:spreadsheet></office:body></office:document-content>'
    )
    path = make_zip(tmp_path / "a.ods", {"content.xml": content})
    assert paras(path) == ["[Animals] Animal: Tortoise; Diet: Cactus."]


def test_odp_one_paragraph_per_slide(tmp_path):
    content = (
        f'<office:document-content {ODF}><office:body><office:presentation>'
        '<draw:page><draw:frame><draw:text-box><text:p>Finches</text:p><text:p>have beaks</text:p></draw:text-box></draw:frame></draw:page>'
        '<draw:page><draw:frame><draw:text-box><text:p>Tortoises</text:p></draw:text-box></draw:frame></draw:page>'
        '</office:presentation></office:body></office:document-content>'
    )
    path = make_zip(tmp_path / "a.odp", {"content.xml": content})
    assert paras(path) == ["Slide 1: Finches. have beaks.", "Slide 2: Tortoises."]


def test_corrupt_office_file_is_an_extract_error(tmp_path):
    for name, files in [("a.docx", {"nope.xml": "<x/>"}), ("b.xlsx", {"xl/workbook.xml": "<broken"})]:
        path = make_zip(tmp_path / name, files)
        try:
            paras(path)
        except extract.ExtractError:
            continue
        raise AssertionError(f"{name} should raise ExtractError")


# --- text-ish formats ------------------------------------------------------


def test_csv_sniffs_delimiter_and_uses_header(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("name;diet\nTortoise;cactus\nFinch;seeds\n")
    assert paras(p) == ["name: Tortoise; diet: cactus. name: Finch; diet: seeds."]


def test_csv_without_header_and_numbers_first_row(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("2024,10,North\n2025,12,South\n")
    assert paras(p) == ["2024 | 10 | North. 2025 | 12 | South."]


def test_many_rows_are_packed_into_several_paragraphs(tmp_path):
    p = tmp_path / "a.tsv"
    p.write_text("name\tnote\n" + "".join(f"row{i}\tsome words about tortoises\n" for i in range(200)))
    out = paras(p)
    assert len(out) > 5
    assert all(len(x) <= extract.TABLE_PARA_CHARS for x in out)


def test_srt_and_vtt_keep_timestamps_and_drop_cue_numbers(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:04,000\nThe tortoises <i>eat</i> cactus.\n\n2\n00:01:05,500 --> 00:01:08,000\nThey rarely drink.\n")
    vtt = tmp_path / "a.vtt"
    vtt.write_text("WEBVTT\n\nNOTE a comment\n\n00:00:02.000 --> 00:00:05.000\nFinches have beaks.\n")
    assert paras(srt) == ["[00:00:01] The tortoises eat cactus. They rarely drink."]
    assert paras(vtt) == ["[00:00:02] Finches have beaks."]


def test_rtf_text_unicode_and_tables(tmp_path):
    rtf = (
        r"{\rtf1\ansi\ansicpg1252{\fonttbl\f0 Times;}{\colortbl;\red0\green0\blue0;}"
        r"{\*\generator Word;}\f0 Gal\'e1pagos tortoises \u8212? big.\par "
        r"Darwin\rquote s notes \{braces\}.\par "
        r"\trowd Island\cell Shell\cell\row}"
    )
    p = tmp_path / "a.rtf"
    p.write_text(rtf)
    assert paras(p) == ["Galápagos tortoises — big.", "Darwin’s notes {braces}.", "Island | Shell"]


def test_eml_headers_and_html_body(tmp_path):
    p = tmp_path / "a.eml"
    p.write_text(
        "From: Bob <bob@example.com>\nSubject: Tortoise update\nContent-Type: text/html; charset=utf-8\n\n"
        "<p>The tortoises ate the cactus.</p>\n"
    )
    assert paras(p) == ["Subject: Tortoise update. From: Bob <bob@example.com>.", "The tortoises ate the cactus."]


def test_xml_text_content(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text('<?xml version="1.0"?><notes><note><title>Finch</title><body>Seeds.</body></note></notes>')
    assert paras(p) == ["Finch Seeds."]


def test_html_table_cells_are_separated(tmp_path):
    p = tmp_path / "a.html"
    p.write_text("<table><tr><th>Island</th><th>Shell</th></tr><tr><td>Hood</td><td>Saddle</td></tr></table>")
    assert paras(p) == ["Island | Shell", "Hood | Saddle"]


def test_code_files_are_searchable_text(tmp_path):
    p = tmp_path / "a.py"
    p.write_text('def tortoise():\n    return "slow"\n')
    assert paras(p) == ['def tortoise(): return "slow"']


def test_every_supported_suffix_has_a_reader(tmp_path):
    for suffix in extract.SUPPORTED_SUFFIXES:
        p = tmp_path / f"x{suffix}"
        p.write_bytes(b"")
        try:
            extract.read_paragraphs(p)
        except extract.ExtractError as exc:
            assert "unsupported" not in str(exc), suffix


def test_spreadsheet_rows_are_not_junk_but_index_entries_are():
    assert not looks_like_junk("[Sales] Month: Jan; Revenue: 1200; Cost: 800. Month: Feb; Revenue: 950; Cost: 700.")
    assert looks_like_junk("Tortoises, 27, 28, 101, 344; shells, 12, 19")
