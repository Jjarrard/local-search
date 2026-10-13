"""Turn files into numbered paragraphs.

Everything downstream works on paragraphs: search hits are paragraphs, the
"does the next one continue?" check walks paragraphs. So this module's one
job is to cut text at natural boundaries and keep each piece small enough
that a tiny model can read it in one go.

Everything except pdf needs only the standard library: docx/xlsx/pptx,
odt/ods/odp and epub are all zip files full of XML underneath.
"""

from __future__ import annotations

import csv
import io
import posixpath
import re
import zipfile
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from itertools import zip_longest
from pathlib import Path
from xml.etree import ElementTree

TEXT_SUFFIXES = {
    # prose and markup
    ".txt", ".text", ".md", ".markdown", ".rst", ".org", ".tex", ".log",
    # config and data that reads fine as-is
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    # code
    ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".java", ".kt", ".swift", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".lua", ".r", ".sh",
    ".zsh", ".bash", ".sql", ".css", ".scss",
}
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}
TABLE_SUFFIXES = {".csv", ".tsv"}
SUBTITLE_SUFFIXES = {".srt", ".vtt"}
ZIP_XML_SUFFIXES = {".epub", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}
SUPPORTED_SUFFIXES = (
    TEXT_SUFFIXES | HTML_SUFFIXES | TABLE_SUFFIXES | SUBTITLE_SUFFIXES | ZIP_XML_SUFFIXES
    | {".xml", ".rtf", ".eml", ".pdf"}
)

# A paragraph bigger than this is split at sentence boundaries. Small enough
# that a 1B model can judge it, big enough to hold a complete thought.
MAX_PARA_CHARS = 1200

# Bump whenever splitting/reading changes, so existing indexes rebuild
# instead of silently keeping paragraphs cut the old way.
EXTRACT_VERSION = 2


class ExtractError(RuntimeError):
    pass


def read_paragraphs(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = _read_text_file(path)
    elif suffix in HTML_SUFFIXES:
        text = html_to_text(_read_text_file(path))
    elif suffix in TABLE_SUFFIXES:
        text = csv_to_text(_read_text_file(path), delimiter="\t" if suffix == ".tsv" else None)
    elif suffix in SUBTITLE_SUFFIXES:
        text = subtitles_to_text(_read_text_file(path))
    elif suffix in ZIP_XML_SUFFIXES:
        text = _read_zip_xml(path, suffix)
    elif suffix == ".xml":
        text = xml_to_text(_read_text_file(path))
    elif suffix == ".rtf":
        text = rtf_to_text(_read_text_file(path))
    elif suffix == ".eml":
        text = eml_to_text(path)
    elif suffix == ".pdf":
        text = pdf_to_text(path)
    else:
        raise ExtractError(f"unsupported file type: {suffix}")
    return split_paragraphs(text)


def _read_zip_xml(path: Path, suffix: str) -> str:
    readers = {
        ".epub": epub_to_text, ".docx": docx_to_text, ".xlsx": xlsx_to_text,
        ".pptx": pptx_to_text, ".odt": odt_to_text, ".ods": ods_to_text, ".odp": odp_to_text,
    }
    try:
        with zipfile.ZipFile(path) as zf:
            return readers[suffix](zf)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, OSError) as exc:
        raise ExtractError(f"not a readable {suffix[1:]} file: {exc}") from exc


def _read_text_file(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")  # never fails; good enough for old txt books


# --- paragraph splitting ---------------------------------------------------

SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_END.split(text)]
    return [p for p in parts if p]


def split_paragraphs(text: str) -> list[str]:
    """Blank lines separate paragraphs; single newlines are just hard wraps.

    Files that put one paragraph per line with no blank lines between (common
    in exported text and PDFs) are detected by their long lines and split
    per line instead.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        joined = " ".join(lines)
        if len(lines) > 1 and _avg(lines) > 150:  # hard-wrapped text is ~70 cols
            pieces = lines
        else:
            pieces = [joined]
        for piece in pieces:
            piece = re.sub(r"\s+", " ", piece).strip()
            if piece:
                out.extend(split_long(piece))
    return out


def _avg(lines: list[str]) -> float:
    return sum(len(ln) for ln in lines) / len(lines)


def split_long(para: str, limit: int = MAX_PARA_CHARS) -> list[str]:
    if len(para) <= limit:
        return [para]
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(para):
        while len(sentence) > limit:  # a monster "sentence" with no full stops
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


# --- html / epub -----------------------------------------------------------

BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "tr", "table", "section", "article", "header", "footer",
    "dd", "dt", "figcaption", "hr",
}
SKIP_TAGS = {"script", "style", "head", "title", "nav"}


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data.replace("\n", " "))


def html_to_text(html: str) -> str:
    parser = _TextCollector()
    parser.feed(html)
    parser.close()
    return _tidy_cells("".join(parser.parts))


def _tidy_cells(text: str) -> str:
    """Drop the dangling separators table cells leave at line edges."""
    return re.sub(r"(?m)^[ \t]*\|[ \t]*|[ \t]*\|[ \t]*$", "", text)


def epub_to_text(zf: zipfile.ZipFile) -> str:
    """An epub is a zip of xhtml files; read them in spine (reading) order."""
    names = zf.namelist()
    order = _epub_spine(zf) or sorted(
        n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))
    )
    texts = []
    for name in order:
        if name in names:
            texts.append(html_to_text(zf.read(name).decode("utf-8", errors="replace")))
    return "\n\n".join(texts)


def _epub_spine(zf: zipfile.ZipFile) -> list[str]:
    try:
        container = ElementTree.fromstring(zf.read("META-INF/container.xml"))
        rootfile = next(el for el in container.iter() if el.tag.endswith("rootfile"))
        opf_path = rootfile.attrib["full-path"]
        opf = ElementTree.fromstring(zf.read(opf_path))
    except (KeyError, StopIteration, ElementTree.ParseError):
        return []
    base = posixpath.dirname(opf_path)
    manifest = {
        el.attrib.get("id"): el.attrib.get("href")
        for el in opf.iter()
        if el.tag.endswith("item")
    }
    spine = []
    for el in opf.iter():
        if el.tag.endswith("itemref"):
            href = manifest.get(el.attrib.get("idref"))
            if href:
                spine.append(posixpath.normpath(posixpath.join(base, href)))
    return spine


# --- shared helpers for structured formats --------------------------------

TABLE_PARA_CHARS = 800


def pack(pieces: list[str], limit: int = TABLE_PARA_CHARS, sep: str = " ") -> list[str]:
    """Glue small pieces (rows, subtitle cues) into paragraphs up to `limit`.

    One spreadsheet row or one subtitle line is too small to be a useful
    search hit - search skips anything under ~100 chars - so they travel
    in groups.
    """
    out: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(sep) + len(piece) > limit:
            out.append(current)
            current = piece
        else:
            current = f"{current}{sep}{piece}" if current else piece
    if current:
        out.append(current)
    return out


def _is_number(cell: str) -> bool:
    return bool(re.fullmatch(r"[-+]?[\d,]*\.?\d+%?", cell.strip()))


def rows_to_paragraphs(rows: list[list[str]], title: str | None = None) -> list[str]:
    """Render table rows as sentences a small model can read.

    With a header row, each row becomes "Name: Bob; Age: 32; City: Leeds." -
    a tiny model can't line up a bare "Bob | 32 | Leeds" with column names
    it saw 40 rows earlier.
    """
    rows = [[" ".join(c.split()) for c in row] for row in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    header = None
    first = rows[0]
    filled = [c for c in first if c]
    if (
        len(rows) > 1
        and len(filled) >= max(2, len(first) // 2)
        and not any(_is_number(c) or len(c) > 40 for c in filled)
    ):
        header, rows = first, rows[1:]
    lines = []
    for row in rows:
        if header:
            line = "; ".join(f"{h or f'column {i + 1}'}: {v}" for i, (h, v) in enumerate(zip_longest(header, row, fillvalue="")) if v)
        else:
            line = " | ".join(c for c in row if c)
        if line:
            lines.append(line.rstrip(".") + ".")
    prefix = f"[{title}] " if title else ""
    return [prefix + p for p in pack(lines)]


def _local(tag: str) -> str:
    """'{namespace}name' -> 'name'. Office XML is namespace soup; we only care about names."""
    return tag.rsplit("}", 1)[-1]


def _attr(el: ElementTree.Element, name: str) -> str | None:
    for key, value in el.attrib.items():
        if _local(key) == name:
            return value
    return None


def _rel_id(el: ElementTree.Element) -> str | None:
    """The r:id relationship attribute - not the plain `id` many elements also carry."""
    for key, value in el.attrib.items():
        if key.startswith("{") and _local(key) == "id":
            return value
    return None


def _xml(zf: zipfile.ZipFile, name: str) -> ElementTree.Element:
    return ElementTree.fromstring(zf.read(name))


def _rels(zf: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """Relationship id -> (zip path, type) for an Office Open XML part."""
    folder, base = posixpath.split(part)
    rels_path = posixpath.join(folder, "_rels", base + ".rels")
    if rels_path not in zf.namelist():
        return {}
    out = {}
    for rel in _xml(zf, rels_path):
        target = rel.get("Target", "")
        if rel.get("TargetMode") == "External":
            continue
        path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
        out[rel.get("Id", "")] = (path, rel.get("Type", ""))
    return out


def _join_sentences(parts: list[str]) -> str:
    """['Title', 'point one.'] -> 'Title. point one.' (no doubled full stops)."""
    out = []
    for part in parts:
        part = " ".join(part.split())
        if part:
            out.append(part if re.search(r"[.!?:;]$", part) else part + ".")
    return " ".join(out)


# --- csv / tsv -------------------------------------------------------------


def csv_to_text(text: str, delimiter: str | None = None) -> str:
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    return "\n\n".join(rows_to_paragraphs(rows))


# --- word / excel / powerpoint (Office Open XML) ---------------------------


def docx_to_text(zf: zipfile.ZipFile) -> str:
    root = _xml(zf, "word/document.xml")
    body = next((el for el in root.iter() if _local(el.tag) == "body"), root)
    out: list[str] = []
    _docx_blocks(body, out)
    return "\n\n".join(out)


def _docx_blocks(parent: ElementTree.Element, out: list[str]) -> None:
    for el in parent:
        tag = _local(el.tag)
        if tag == "p":
            text = _docx_paragraph(el)
            if text:
                out.append(text)
        elif tag == "tbl":
            rows = []
            for tr in el.iter():
                if _local(tr.tag) == "tr":
                    rows.append([
                        " ".join(_docx_paragraph(p) for p in tc.iter() if _local(p.tag) == "p")
                        for tc in tr if _local(tc.tag) == "tc"
                    ])
            out.extend(rows_to_paragraphs(rows))
        elif tag in ("sdt", "sdtContent", "customXml", "ins", "smartTag"):
            _docx_blocks(el, out)  # content controls and tracked insertions wrap real paragraphs


def _docx_paragraph(p: ElementTree.Element) -> str:
    parts = []
    for el in p.iter():
        tag = _local(el.tag)
        if tag == "t":
            parts.append(el.text or "")
        elif tag in ("tab", "br", "cr"):
            parts.append(" ")
        elif tag == "noBreakHyphen":
            parts.append("-")
    return " ".join("".join(parts).split())


def xlsx_to_text(zf: zipfile.ZipFile) -> str:
    shared: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        for si in _xml(zf, "xl/sharedStrings.xml"):
            # Plain <t>, or rich-text runs <r><t>. Skips <rPh> phonetic guides,
            # which repeat the text in another script.
            parts = []
            for child in si:
                if _local(child.tag) == "t":
                    parts.append(child.text or "")
                elif _local(child.tag) == "r":
                    parts.extend(t.text or "" for t in child if _local(t.tag) == "t")
            shared.append("".join(parts))
    workbook = _xml(zf, "xl/workbook.xml")
    rels = _rels(zf, "xl/workbook.xml")
    out: list[str] = []
    for sheet in (el for el in workbook.iter() if _local(el.tag) == "sheet"):
        target = rels.get(_rel_id(sheet) or "")
        if not target or target[0] not in zf.namelist():
            continue
        rows = []
        for row in (el for el in _xml(zf, target[0]).iter() if _local(el.tag) == "row"):
            cells: dict[int, str] = {}
            for c in row:
                if _local(c.tag) != "c":
                    continue
                col = _column_index(c.get("r", ""))
                cells[col if col is not None else len(cells)] = _xlsx_value(c, shared)
            if cells:
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
        out.extend(rows_to_paragraphs(rows, title=sheet.get("name")))
    return "\n\n".join(out)


def _column_index(ref: str) -> int | None:
    letters = re.match(r"[A-Z]+", ref.upper())
    if not letters:
        return None
    n = 0
    for ch in letters.group():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _xlsx_value(c: ElementTree.Element, shared: list[str]) -> str:
    kind = c.get("t")
    v = next((el.text or "" for el in c if _local(el.tag) == "v"), "")
    if kind == "s":
        try:
            return shared[int(v)]
        except (ValueError, IndexError):
            return ""
    if kind == "inlineStr":
        return "".join(t.text or "" for t in c.iter() if _local(t.tag) == "t")
    if kind == "b":
        return "TRUE" if v == "1" else "FALSE"
    return v


def pptx_to_text(zf: zipfile.ZipFile) -> str:
    """One paragraph per slide - 'Slide 3: Title. Bullet. Bullet. Notes: ...'"""
    pres = _xml(zf, "ppt/presentation.xml")
    rels = _rels(zf, "ppt/presentation.xml")
    slides = [
        rels[rid][0] for el in pres.iter()
        if _local(el.tag) == "sldId" and (rid := _rel_id(el)) in rels
    ]
    if not slides:  # fall back to file order: slide1, slide2, ... slide10
        slides = sorted(
            (n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)),
        )
    out = []
    for number, slide in enumerate(slides, 1):
        if slide not in zf.namelist():
            continue
        text = _join_sentences(_drawing_paragraphs(_xml(zf, slide)))
        notes = [path for path, kind in _rels(zf, slide).values() if kind.endswith("/notesSlide")]
        if notes and notes[0] in zf.namelist():
            # the notes page repeats the slide number as text; drop bare numbers
            note_text = _join_sentences([t for t in _drawing_paragraphs(_xml(zf, notes[0])) if not t.strip().isdigit()])
            if note_text:
                text = f"{text} Notes: {note_text}".strip()
        if text:
            out.append(f"Slide {number}: {text}")
    return "\n\n".join(out)


def _drawing_paragraphs(root: ElementTree.Element) -> list[str]:
    paras = []
    for p in root.iter():
        if _local(p.tag) == "p":
            text = "".join(t.text or "" for t in p.iter() if _local(t.tag) == "t")
            if text.strip():
                paras.append(text)
    return paras


# --- libreoffice (OpenDocument) --------------------------------------------


def _odf_text(el: ElementTree.Element) -> str:
    """Text of an ODF element, honouring <text:s/> (spaces), tabs and line breaks."""
    parts = [el.text or ""]
    for child in el:
        tag = _local(child.tag)
        if tag == "s":
            parts.append(" " * int(_attr(child, "c") or 1))
        elif tag in ("tab", "line-break"):
            parts.append(" ")
        elif tag not in ("annotation", "note-citation"):
            parts.append(_odf_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def odt_to_text(zf: zipfile.ZipFile) -> str:
    root = _xml(zf, "content.xml")
    out = []
    for el in root.iter():
        if _local(el.tag) in ("p", "h"):
            text = " ".join(_odf_text(el).split())
            if text:
                out.append(text)
    return "\n\n".join(out)


def odp_to_text(zf: zipfile.ZipFile) -> str:
    root = _xml(zf, "content.xml")
    out = []
    pages = [el for el in root.iter() if _local(el.tag) == "page"]
    for number, page in enumerate(pages, 1):
        paras = [_odf_text(el) for el in page.iter() if _local(el.tag) in ("p", "h")]
        text = _join_sentences(paras)
        if text:
            out.append(f"Slide {number}: {text}")
    return "\n\n".join(out)


def ods_to_text(zf: zipfile.ZipFile) -> str:
    root = _xml(zf, "content.xml")
    out: list[str] = []
    for table in (el for el in root.iter() if _local(el.tag) == "table"):
        rows = []
        for row in (el for el in table.iter() if _local(el.tag) == "table-row"):
            cells: list[str] = []
            for cell in row:
                if _local(cell.tag) not in ("table-cell", "covered-table-cell"):
                    continue
                text = " ".join(_odf_text(p) for p in cell if _local(p.tag) == "p").strip()
                # Empty cells are often "repeated 16000 times" to pad to the sheet edge.
                repeat = int(_attr(cell, "number-columns-repeated") or 1)
                cells.extend([text] * (min(repeat, 100) if text else min(repeat, 1000)))
            while cells and not cells[-1]:
                cells.pop()
            if cells:
                rows.extend([cells] * min(int(_attr(row, "number-rows-repeated") or 1), 100))
        out.extend(rows_to_paragraphs(rows, title=_attr(table, "name")))
    return "\n\n".join(out)


# --- xml / rtf / email / subtitles -----------------------------------------


def xml_to_text(text: str) -> str:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return html_to_text(text)
    pieces = [" ".join(t.split()) for t in root.itertext()]
    return "\n\n".join(pack([p for p in pieces if p], limit=MAX_PARA_CHARS))


RTF_TOKEN = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)", re.I | re.S)
# Groups that hold fonts, colours, metadata and pictures rather than text.
RTF_SKIP = {
    "fonttbl", "colortbl", "stylesheet", "info", "pict", "header", "footer", "headerl",
    "headerr", "footerl", "footerr", "listtable", "listoverridetable", "rsidtbl",
    "generator", "xmlnstbl", "themedata", "colorschememapping", "latentstyles", "datastore",
    "object", "fldinst", "filetbl", "revtbl", "pgdsctbl", "operator", "author", "title",
}
RTF_CHARS = {
    "par": "\n\n", "line": "\n", "sect": "\n\n", "page": "\n\n", "tab": " ", "cell": " | ",
    "row": "\n\n", "emdash": "—", "endash": "–", "bullet": "•", "lquote": "‘", "rquote": "’",
    "ldblquote": "“", "rdblquote": "”", "emspace": " ", "enspace": " ", "qmspace": " ",
}


def rtf_to_text(rtf: str) -> str:
    """Strip RTF down to its text. Enough for what Word, TextEdit and
    WordPad write; not a full RTF implementation."""
    out: list[str] = []
    stack: list[tuple[bool, int]] = []
    skipping = False
    uc = 1          # characters to skip after a \uN unicode escape
    pending = 0
    for m in RTF_TOKEN.finditer(rtf):
        word, arg, hexcode, symbol, brace, char = m.groups()
        if brace == "{":
            stack.append((skipping, uc))
        elif brace == "}":
            skipping, uc = stack.pop() if stack else (False, 1)
        elif symbol:
            pending = 0
            if symbol == "*":
                skipping = True
            elif not skipping:
                if symbol in "\\{}":
                    out.append(symbol)
                elif symbol == "~":
                    out.append(" ")
                elif symbol == "_":
                    out.append("-")
                elif symbol in "\r\n":
                    out.append("\n\n")
        elif word:
            pending = 0
            word = word.lower()
            if word in RTF_SKIP:
                skipping = True
            elif word == "uc":
                uc = int(arg or 1)
            elif skipping:
                continue
            elif word == "u" and arg:
                code = int(arg)
                out.append(chr(code + 65536 if code < 0 else code))
                pending = uc
            elif word in RTF_CHARS:
                out.append(RTF_CHARS[word])
        elif hexcode:
            if pending:
                pending -= 1
            elif not skipping:
                out.append(bytes([int(hexcode, 16)]).decode("cp1252", errors="replace"))
        elif char:
            if pending:
                pending -= 1
            elif not skipping:
                out.append(char)
    return _tidy_cells("".join(out))


def eml_to_text(path: Path) -> str:
    with path.open("rb") as fh:
        msg = BytesParser(policy=policy.default).parse(fh)
    head = _join_sentences([f"{h}: {msg[h]}" for h in ("Subject", "From", "To", "Date") if msg[h]])
    body = msg.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        try:
            text = body.get_content()
        except (LookupError, ValueError):
            text = (body.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
        if body.get_content_subtype() == "html":
            text = html_to_text(text)
    return f"{head}\n\n{text}"


def subtitles_to_text(text: str) -> str:
    """srt / vtt -> paragraphs of ~800 chars, each starting with its timestamp,
    so a citation tells you where in the video to look."""
    cues: list[tuple[str | None, str]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines or lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        start, words = None, []
        for line in lines:
            if "-->" in line:
                start = line.split("-->")[0].strip().split(".")[0].split(",")[0]
            elif not line.isdigit():
                words.append(re.sub(r"<[^>]+>|\{\\[^}]*\}", "", line))
        if words:
            cues.append((start, " ".join(words)))
    paras, current, stamp = [], "", None
    for start, words in cues:
        if current and len(current) + 1 + len(words) > TABLE_PARA_CHARS:
            paras.append(f"[{stamp}] {current}" if stamp else current)
            current, stamp = "", None
        if not current:
            stamp = start
        current = f"{current} {words}".strip()
    if current:
        paras.append(f"[{stamp}] {current}" if stamp else current)
    return "\n\n".join(paras)


# --- pdf -------------------------------------------------------------------


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractError("pdf support needs pypdf: .venv/bin/pip install pypdf") from exc
    try:
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 - pypdf raises a zoo of error types
        raise ExtractError(f"could not read pdf: {exc}") from exc
