import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    # Never touch the real ~/.localsearch from tests.
    monkeypatch.setenv("LOCALSEARCH_CONFIG_DIR", str(tmp_path / "cfg"))
