"""Persisted user preferences (model, server, search knobs).

Lives in ~/.localsearch/config.json, alongside the per-folder search indexes.
Every read is defensive: a corrupt or hand-edited config degrades to defaults
instead of crashing the app on startup.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def config_dir() -> Path:
    override = os.environ.get("LOCALSEARCH_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".localsearch"


def config_path() -> Path:
    return config_dir() / "config.json"


@dataclass
class Config:
    """User preferences. Field names double as the on-disk keys."""

    model: str | None = None
    base_url: str = "http://127.0.0.1:1338/v1"
    api_key: str = "not-needed"
    folder: str | None = None           # last folder searched
    top: int = 10                       # search hits to look at per question
    max_grow: int = 3                   # max paragraphs to follow past a hit
    context_size: int = 4096            # model context window, in tokens
    no_think: bool = True               # ask reasoning models to skip <think>
    verbose: bool = False
    recent_models: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "Config":
        path = config_path()
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in raw.items() if k in known}
        try:
            cfg = cls(**clean)
        except TypeError:
            return cls()
        if not isinstance(cfg.recent_models, list):
            cfg.recent_models = []
        return cfg

    def save(self) -> None:
        path = config_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2))
            tmp.replace(path)  # atomic: a crash mid-write can't corrupt the config
        except OSError:
            pass  # preferences are a convenience, never worth crashing over

    def remember_model(self, model: str) -> None:
        if not model:
            return
        recents = [m for m in self.recent_models if m != model]
        recents.insert(0, model)
        self.recent_models = recents[:10]

    def order_models(self, available: list[str]) -> list[str]:
        """Most-recently-used first, so the picker opens on what you actually use."""
        recent = [m for m in self.recent_models if m in available]
        rest = sorted(m for m in available if m not in recent)
        return recent + rest
