from rich.console import Console

from localsearch import cli, servers
from localsearch.client import ChatClient
from localsearch.config import Config


def test_model_size_reads_common_ids():
    assert servers.model_size("qwen3:4b") == 4
    assert servers.model_size("qwen2.5-coder:1.5b") == 1.5
    assert servers.model_size("Jan-v3.5-4B-Q4_K_XL") == 4
    assert servers.model_size("DeepSeek-R1-Distill-Qwen-14B-Q5_K_M") == 14
    assert servers.model_size("gemma4:e4b") == 4
    assert servers.model_size("llama3.2:latest") is None


def test_pick_default_prefers_the_largest_small_model():
    assert servers.pick_default_model(["qwen2.5-coder:14b", "qwen3:4b", "gemma3:1b"]) == "qwen3:4b"


def test_pick_default_falls_back_to_smallest_bigger_model_then_first():
    assert servers.pick_default_model(["big-14b", "mid-7b"]) == "mid-7b"
    assert servers.pick_default_model(["llama3.2:latest", "mystery"]) == "llama3.2:latest"
    assert servers.pick_default_model([]) is None


def test_find_servers_keeps_only_reachable_ones(monkeypatch):
    answers = {"http://127.0.0.1:11434/v1": ["qwen3:4b"], "http://127.0.0.1:1234/v1": []}
    monkeypatch.setattr(servers, "probe", lambda url, key="": answers.get(url))
    found = servers.find_servers()
    assert [(s.name, s.models) for s in found] == [("Ollama", ["qwen3:4b"]), ("LM Studio", [])]


def test_connect_switches_to_another_running_server(monkeypatch):
    monkeypatch.setattr(servers, "probe", lambda url, key="", timeout=0: None)
    monkeypatch.setattr(
        servers, "find_servers",
        lambda extra=(), api_key="": [servers.Server("Ollama", "http://127.0.0.1:11434/v1", ["qwen3:4b"])],
    )
    cfg = Config()
    client = ChatClient(base_url=cfg.base_url)
    models = cli.connect(Console(quiet=True), cfg, client, explicit_url=False)
    assert models == ["qwen3:4b"]
    assert cfg.base_url == client.base_url == "http://127.0.0.1:11434/v1"


def test_connect_respects_an_explicit_url(monkeypatch):
    monkeypatch.setattr(servers, "probe", lambda url, key="", timeout=0: None)
    monkeypatch.setattr(servers, "find_servers", lambda **_: (_ for _ in ()).throw(AssertionError("searched")))
    cfg = Config(base_url="http://example:9/v1")
    assert cli.connect(Console(quiet=True), cfg, ChatClient(base_url=cfg.base_url), explicit_url=True) == []
    assert cfg.base_url == "http://example:9/v1"
