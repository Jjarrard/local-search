"""Finding a model server, picking a model, and getting one if there's none.

localsearch needs any OpenAI-compatible server with a chat model loaded. The
common local ones all listen on a fixed default port, so instead of making
people know and type a URL, we knock on each of those doors.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from .client import is_chat_model

# (name, OpenAI-compatible base URL), in the order we prefer them.
KNOWN_SERVERS = (
    ("Jan", "http://127.0.0.1:1338/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
)

# What `setup` offers to download through Ollama when there's no model yet.
# Small enough for a laptop, good at short closed questions, Apache 2.0.
RECOMMENDED_OLLAMA_MODEL = "qwen3:4b"


@dataclass
class Server:
    name: str
    base_url: str
    models: list[str]  # chat models only


def probe(base_url: str, api_key: str = "not-needed", timeout: float = 1.5) -> list[str] | None:
    """Chat models on the server at base_url, or None if nothing answers there."""
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    return [m for m in ids if isinstance(m, str) and is_chat_model(m)]


def name_for(base_url: str) -> str:
    for name, url in KNOWN_SERVERS:
        if url.rstrip("/") == base_url.rstrip("/"):
            return name
    return "server"


def find_servers(extra: tuple[str, ...] = (), api_key: str = "not-needed") -> list[Server]:
    """Every reachable server: `extra` URLs first, then the well-known ports."""
    urls: list[str] = []
    for url in (*extra, *(u for _, u in KNOWN_SERVERS)):
        if url and url.rstrip("/") not in (u.rstrip("/") for u in urls):
            urls.append(url)
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        results = list(pool.map(lambda u: probe(u, api_key), urls))
    return [Server(name_for(u), u, models) for u, models in zip(urls, results) if models is not None]


_SIZE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b(?![a-z])", re.IGNORECASE)


def model_size(name: str) -> float | None:
    """Billions of parameters from a model id like 'qwen3:4b' or 'Jan-v3.5-4B-Q4'."""
    sizes = [float(m) for m in _SIZE.findall(name.replace(":", " ").replace("-", " ").replace("_", " "))]
    sizes = [s for s in sizes if 0.1 <= s <= 1000]
    return sizes[0] if sizes else None


def pick_default_model(models: list[str]) -> str | None:
    """The model to use when the user hasn't chosen one.

    The pipeline is built for 1-4B models: they answer its YES/NO questions
    quickly, and bigger ones only make every question slower. So prefer the
    largest model in that range, then the smallest one above it, then
    whatever is left.
    """
    if not models:
        return None
    sized = [(model_size(m), m) for m in models]
    in_range = [(s, m) for s, m in sized if s is not None and 1 <= s <= 4.5]
    if in_range:
        return max(in_range)[1]
    larger = [(s, m) for s, m in sized if s is not None and s > 4.5]
    if larger:
        return min(larger)[1]
    return models[0]


def ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def ollama_pull(model: str, out: Callable[[str], None] = print) -> bool:
    """Download a model with the ollama CLI, showing its progress. True on success."""
    try:
        proc = subprocess.run(["ollama", "pull", model], check=False)
    except OSError as exc:
        out(f"could not run ollama: {exc}")
        return False
    return proc.returncode == 0
