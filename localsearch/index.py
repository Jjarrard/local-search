"""Full-text index of a folder, one row per paragraph.

SQLite's built-in FTS5 does the heavy lifting: BM25 ranking, porter
stemming ("volcanoes" finds "volcano"), and it handles a shelf of books in
milliseconds. No embeddings, no model, nothing to install.

The index lives in ~/.localsearch/indexes/, not inside the searched folder,
so a read-only folder (or one you don't want littered) still works.
Re-indexing only re-reads files whose size or mtime changed.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from . import extract
from .config import config_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    n_paras INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS paras (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    n INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS paras_file_n ON paras(file_id, n);
CREATE VIRTUAL TABLE IF NOT EXISTS paras_fts USING fts5(
    text, content='paras', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS paras_ai AFTER INSERT ON paras BEGIN
    INSERT INTO paras_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS paras_ad AFTER DELETE ON paras BEGIN
    INSERT INTO paras_fts(paras_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


@dataclass
class Hit:
    file_id: int
    path: str       # relative to the indexed folder
    n: int          # paragraph number within the file (0-based)
    text: str
    score: float    # bm25: lower is better


@dataclass
class RefreshStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def db_path_for(folder: Path) -> Path:
    folder = folder.resolve()
    digest = hashlib.sha1(str(folder).encode()).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", folder.name)[:40] or "root"
    return config_dir() / "indexes" / f"{slug}-{digest}.db"


def iter_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        rel_parts = path.relative_to(folder).parts
        if any(p.startswith(".") for p in rel_parts):
            continue  # hidden files and folders (.git, .venv, ...)
        if path.is_file() and path.suffix.lower() in extract.SUPPORTED_SUFFIXES:
            yield path


def fts_query(terms: Iterable[str], op: str = "OR") -> str:
    """Build a safe FTS5 query. Each term becomes a quoted phrase, so user
    text can never be parsed as FTS syntax (AND, NEAR, *, column filters)."""
    parts = []
    for term in terms:
        words = re.findall(r"\w+", term.lower())
        if words:
            parts.append('"' + " ".join(words) + '"')
    return f" {op} ".join(dict.fromkeys(parts))


class Index:
    def __init__(self, folder: Path, db_path: Path | None = None):
        self.folder = folder.resolve()
        self.db_path = db_path or db_path_for(self.folder)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        if self.conn.execute("PRAGMA user_version").fetchone()[0] != extract.EXTRACT_VERSION:
            with self.conn:  # extraction changed since this index was built: start over
                self.conn.execute("DELETE FROM paras")
                self.conn.execute("DELETE FROM files")
                self.conn.execute(f"PRAGMA user_version = {int(extract.EXTRACT_VERSION)}")

    def close(self) -> None:
        self.conn.close()

    def refresh(self, progress: Callable[[int, int, str], None] | None = None) -> RefreshStats:
        stats = RefreshStats()
        known = {
            row[0]: (row[1], row[2], row[3])
            for row in self.conn.execute("SELECT path, id, mtime, size FROM files")
        }
        on_disk = list(iter_files(self.folder))
        seen: set[str] = set()
        for i, path in enumerate(on_disk, 1):
            rel = path.relative_to(self.folder).as_posix()
            seen.add(rel)
            st = path.stat()
            old = known.get(rel)
            if old and old[1] == st.st_mtime and old[2] == st.st_size:
                stats.unchanged += 1
                continue
            if progress:
                progress(i, len(on_disk), rel)
            try:
                paras = extract.read_paragraphs(path)
            except (extract.ExtractError, OSError) as exc:
                stats.failed.append((rel, str(exc)))
                continue
            with self.conn:
                if old:
                    self._drop_file(old[0])
                cur = self.conn.execute(
                    "INSERT INTO files(path, mtime, size, n_paras) VALUES (?, ?, ?, ?)",
                    (rel, st.st_mtime, st.st_size, len(paras)),
                )
                self.conn.executemany(
                    "INSERT INTO paras(file_id, n, text) VALUES (?, ?, ?)",
                    [(cur.lastrowid, n, text) for n, text in enumerate(paras)],
                )
            if old:
                stats.updated += 1
            else:
                stats.added += 1
        for rel, (file_id, _, _) in known.items():
            if rel not in seen:
                with self.conn:
                    self._drop_file(file_id)
                stats.removed += 1
        return stats

    def _drop_file(self, file_id: int) -> None:
        self.conn.execute("DELETE FROM paras WHERE file_id = ?", (file_id,))
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def search(self, query: str, limit: int = 10, min_chars: int = 0) -> list[Hit]:
        """Best paragraphs for an FTS query. `min_chars` drops title lines,
        headings and back-of-book index entries, which BM25 over-ranks
        because they're short and dense with the search words."""
        if not query:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT p.file_id, f.path, p.n, p.text, bm25(paras_fts) AS score
                FROM paras_fts
                JOIN paras p ON p.id = paras_fts.rowid
                JOIN files f ON f.id = p.file_id
                WHERE paras_fts MATCH ? AND length(p.text) >= ?
                ORDER BY score
                LIMIT ?
                """,
                (query, min_chars, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # a malformed query is "no results", never a crash
        return [Hit(*row) for row in rows]

    def paragraph(self, file_id: int, n: int) -> str | None:
        if n < 0:
            return None
        row = self.conn.execute(
            "SELECT text FROM paras WHERE file_id = ? AND n = ?", (file_id, n)
        ).fetchone()
        return row[0] if row else None

    def counts(self) -> tuple[int, int]:
        files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        paras = self.conn.execute("SELECT COUNT(*) FROM paras").fetchone()[0]
        return files, paras
