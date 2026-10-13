"""Turn files into numbered paragraphs.

Everything downstream works on paragraphs: search hits are paragraphs, the
"does the next one continue?" check walks paragraphs. So this module's one
job is to cut text at natural boundaries and keep each piece small enough
that a tiny model can read it in one go.

txt / md / html / epub need only the standard library. pdf needs pypdf.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".rst", ".org"}
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | {".epub", ".pdf"}

# A paragraph bigger than this is split at sentence boundaries. Small enough
# that a 1B model can judge it, big enough to hold a complete thought.
MAX_PARA_CHARS = 1200

# Bump whenever splitting/reading changes, so existing indexes rebuild
# instead of silently keeping paragraphs cut the old way.
EXTRACT_VERSION = 1


class ExtractError(RuntimeError):
    pass


def read_paragraphs(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = _read_text_file(path)
    elif suffix in HTML_SUFFIXES:
        text = html_to_text(_read_text_file(path))
    elif suffix == ".epub":
        text = epub_to_text(path)
    elif suffix == ".pdf":
        text = pdf_to_text(path)
    else:
        raise ExtractError(f"unsupported file type: {suffix}")
    return split_paragraphs(text)


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
    return "".join(parser.parts)


def epub_to_text(path: Path) -> str:
    """An epub is a zip of xhtml files; read them in spine (reading) order."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            order = _epub_spine(zf) or sorted(
                n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))
            )
            texts = []
            for name in order:
                if name in names:
                    texts.append(html_to_text(zf.read(name).decode("utf-8", errors="replace")))
            return "\n\n".join(texts)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ExtractError(f"not a readable epub: {exc}") from exc


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
