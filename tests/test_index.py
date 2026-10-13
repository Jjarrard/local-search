import os

from localsearch.index import Index, fts_query


def make_folder(tmp_path):
    folder = tmp_path / "books"
    folder.mkdir()
    (folder / "a.txt").write_text("Volcanoes erupt with lava.\n\nThe weather was fine.\n\nTortoises are slow.")
    (folder / "b.md").write_text("# Notes\n\nA volcano is a mountain.")
    (folder / ".hidden").mkdir()
    (folder / ".hidden" / "secret.txt").write_text("volcano secret")
    (folder / "image.png").write_bytes(b"\x89PNG")
    return folder


def test_index_and_search_with_stemming(tmp_path):
    idx = Index(make_folder(tmp_path), tmp_path / "i.db")
    stats = idx.refresh()
    assert stats.added == 2  # hidden folder and png skipped
    hits = idx.search(fts_query(["volcano"]))
    assert {h.path for h in hits} == {"a.txt", "b.md"}  # "volcano" also finds "Volcanoes"


def test_paragraph_lookup_by_number(tmp_path):
    idx = Index(make_folder(tmp_path), tmp_path / "i.db")
    idx.refresh()
    hit = idx.search(fts_query(["tortoises"]))[0]
    assert hit.n == 2
    assert idx.paragraph(hit.file_id, 1) == "The weather was fine."
    assert idx.paragraph(hit.file_id, 3) is None
    assert idx.paragraph(hit.file_id, -1) is None


def test_refresh_only_rereads_changed_files_and_drops_deleted(tmp_path):
    folder = make_folder(tmp_path)
    idx = Index(folder, tmp_path / "i.db")
    idx.refresh()
    (folder / "a.txt").write_text("Completely new text about penguins.")
    os.utime(folder / "a.txt", (1, 1))
    (folder / "b.md").unlink()
    stats = idx.refresh()
    assert (stats.updated, stats.removed, stats.added) == (1, 1, 0)
    assert idx.search(fts_query(["volcano"])) == []
    assert idx.search(fts_query(["penguins"]))[0].path == "a.txt"
    assert idx.refresh().changed is False


def test_min_chars_drops_short_hits(tmp_path):
    idx = Index(make_folder(tmp_path), tmp_path / "i.db")
    idx.refresh()
    assert idx.search(fts_query(["volcano"]), min_chars=25)[0].path == "a.txt"
    assert len(idx.search(fts_query(["volcano"]), min_chars=1000)) == 0


def test_fts_query_neutralises_syntax():
    q = fts_query(['NEAR(x', 'a" OR "b', "roman empire", "roman empire"])
    assert q == '"near x" OR "a or b" OR "roman empire"'
    assert fts_query(["!!!"]) == ""


def test_hostile_query_returns_nothing_instead_of_crashing(tmp_path):
    idx = Index(make_folder(tmp_path), tmp_path / "i.db")
    idx.refresh()
    assert idx.search('"unclosed') == []


def test_extractor_version_change_rebuilds_index(tmp_path, monkeypatch):
    from localsearch import extract

    folder = make_folder(tmp_path)
    Index(folder, tmp_path / "i.db").refresh()
    monkeypatch.setattr(extract, "EXTRACT_VERSION", extract.EXTRACT_VERSION + 1)
    idx = Index(folder, tmp_path / "i.db")
    assert idx.counts() == (0, 0)
    assert idx.refresh().added == 2
    assert Index(folder, tmp_path / "i.db").counts()[0] == 2  # same version: kept
