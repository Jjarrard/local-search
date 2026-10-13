"""Tab completion for the REPL.

`/model` (then Tab) cycles through the models the server has loaded,
`/folder` completes paths, and any other `/` command completes by name.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable

SLASH_COMMANDS = [
    "/help", "/status", "/model", "/folder", "/reindex", "/sources", "/top",
    "/grow", "/llm", "/verbose", "/quit", "/exit",
]
MODEL_COMMANDS = ("/model",)
PATH_COMMANDS = ("/folder",)


def candidates(buffer: str, text: str, models: list[str]) -> list[str]:
    """Given the full input line so far and the word currently being
    completed, return sorted completion candidates."""
    if buffer.startswith(MODEL_COMMANDS):
        options = models
    elif buffer.startswith(PATH_COMMANDS) and " " in buffer:
        expanded = os.path.expanduser(text)
        options = []
        for match in glob.glob(expanded + "*"):
            if os.path.isdir(match):
                shown = match if not text.startswith("~") else "~" + match[len(os.path.expanduser("~")):]
                options.append(shown + "/")
        return sorted(options)
    elif buffer.startswith("/") and " " not in buffer:
        options = SLASH_COMMANDS
    else:
        options = []
    return sorted(o for o in options if o.startswith(text))


def install(get_models: Callable[[], list[str]]) -> None:
    """Wire up readline tab completion. No-op if readline isn't available."""
    try:
        import readline
    except ImportError:
        return

    # Model ids often contain "/" (e.g. "bartowski/DeepSeek-R1-..."), which
    # is a default word-delimiter. Restrict delimiters to whitespace.
    readline.set_completer_delims(" \t\n")

    def completer(text: str, state: int) -> str | None:
        buffer = readline.get_line_buffer()
        options = candidates(buffer, text, get_models())
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    is_libedit = "libedit" in (readline.__doc__ or "")
    bind = "bind ^I rl_complete" if is_libedit else "tab: menu-complete"
    try:
        readline.parse_and_bind(bind)
    except Exception:  # noqa: BLE001 - tab completion is a nicety, never fatal
        pass
