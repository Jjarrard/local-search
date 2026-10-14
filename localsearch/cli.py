"""Terminal app for localsearch: ask questions about a folder of files."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import completion, servers, ui
from .client import APIConnectionError, ChatClient
from .config import Config
from .index import Index
from .pipeline import TLDR_REPLY_TOKENS, Gathered, Searcher, tldr_prompt

HELP = """\
type a question to search the folder and get a TLDR.

slash commands:
  /help                 show this
  /status               show model, folder, and settings
  /model [name]         pick the model - no name opens an arrow-key picker
  /folder <path>        search a different folder (indexes it first)
  /reindex              re-scan the folder for new or changed files
  /sources              show the full passages behind the last answer
  /top <n>              how many search hits to look at (default 10)
  /grow <n>             max paragraphs to follow past a hit (default 3)
  /llm [on|off]         off = search + extracts only, no model calls
  /verbose [on|off]     show every grow/filter decision
  /quit, /exit          leave
"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="localsearch",
        description=__doc__,
        epilog="first time? run `localsearch setup` to find a model server and pick a folder.",
    )
    p.add_argument("question", nargs="*", help="ask once and exit instead of starting the REPL")
    p.add_argument("--folder", default=None, help="folder to search (remembered between runs; default: current dir)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL (Jan, Ollama, ...)")
    p.add_argument("--model", default=None, help="model id (see /v1/models); remembered between runs")
    p.add_argument("--top", type=int, default=None, help="search hits to look at per question")
    p.add_argument("--max-grow", type=int, default=None, help="max paragraphs to follow past a hit")
    p.add_argument("--context-size", type=int, default=None, help="model context window in tokens (default 4096)")
    p.add_argument("--no-llm", action="store_true", help="no model at all: search + extracts only")
    p.add_argument("--think", action="store_true", help="let reasoning models think (slower; off by default)")
    p.add_argument("--verbose", action="store_true", help="show every grow/filter decision")
    return p


@dataclass
class App:
    console: Console
    cfg: Config
    client: ChatClient
    index: Index
    available: list[str]
    use_llm: bool
    verbose: bool
    last: Gathered | None = None

    def searcher(self) -> Searcher:
        return Searcher(
            index=self.index,
            ask=self.client.ask if self.use_llm else None,
            top=self.cfg.top,
            max_grow=self.cfg.max_grow,
            context_size=self.cfg.context_size,
            log=lambda m: self.console.print(f"[dim]· {escape(m)}[/dim]"),
            verbose=(lambda m: self.console.print(f"[dim]    {escape(m)}[/dim]")) if self.verbose else (lambda _m: None),
        )


def open_index(console: Console, folder: Path) -> Index:
    index = Index(folder)
    with ui.status(console, f"indexing {folder}") as st:
        def progress(i: int, total: int, rel: str) -> None:
            if st is not None:
                st.update(f"[dim]indexing {i}/{total}: {escape(rel)}[/dim]")
        stats = index.refresh(progress)
    files, paras = index.counts()
    changes = []
    if stats.added:
        changes.append(f"{stats.added} new")
    if stats.updated:
        changes.append(f"{stats.updated} changed")
    if stats.removed:
        changes.append(f"{stats.removed} removed")
    note = f" ({', '.join(changes)})" if changes else ""
    console.print(f"[dim]index: {files} files, {paras} paragraphs{note}[/dim]")
    for rel, err in stats.failed:
        console.print(f"[yellow]skipped {escape(rel)}: {escape(err)}[/yellow]")
    if files == 0:
        console.print("[yellow]no searchable files here (see README for supported types)[/yellow]")
    return index


def answer(app: App, question: str) -> None:
    console = app.console
    searcher = app.searcher()
    try:
        gathered = searcher.gather(question)
    except KeyboardInterrupt:
        console.print("[yellow]stopped[/yellow]")
        return
    app.last = gathered
    if not gathered.passages:
        console.print("[yellow]nothing found.[/yellow] try different words, or /top to look wider.")
        return

    console.print()
    if app.use_llm:
        console.print("[bold]TLDR[/bold]")
        prompt = tldr_prompt(question, gathered.passages)
        try:
            with ui.status(console, f"writing TLDR ({app.client.model})"):
                stream = app.client.stream_answer(prompt, max_tokens=TLDR_REPLY_TOKENS)
                first = next(stream, "")  # spinner runs until the first visible text
            console.print(first, end="", markup=False, highlight=False)
            for piece in stream:
                console.print(piece, end="", markup=False, highlight=False)
            console.print()
        except APIConnectionError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]stopped[/yellow]")
        console.print()
        console.print("[bold]sources[/bold]")
        for i, p in enumerate(gathered.passages, 1):
            preview = " ".join(p.extract)
            preview = preview if len(preview) <= 160 else preview[:157] + "..."
            console.print(f"[cyan][{i}][/cyan] [dim]{escape(p.label)}[/dim]  {escape(preview)}")
        console.print(f"[dim]{searcher.calls} model calls · /sources for full passages[/dim]")
    else:
        for i, p in enumerate(gathered.passages, 1):
            console.print(f"[cyan][{i}][/cyan] [bold]{escape(p.label)}[/bold]")
            for s in p.extract:
                console.print(f"    {escape(s)}")
        console.print("[dim]/sources for full passages[/dim]")
    console.print()


def print_sources(app: App) -> None:
    if not app.last or not app.last.passages:
        app.console.print("[dim](no answer yet)[/dim]")
        return
    for i, p in enumerate(app.last.passages, 1):
        app.console.rule(f"[{i}] {p.label}", align="left", style="dim")
        app.console.print(escape(p.text))
        app.console.print()


def print_status(app: App) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    files, paras = app.index.counts()
    table.add_row("folder", str(app.index.folder))
    table.add_row("index", f"{files} files, {paras} paragraphs")
    table.add_row("model", app.client.model if app.use_llm else "(off - search + extracts only)")
    table.add_row("server", app.client.base_url)
    table.add_row("hits / grow", f"top {app.cfg.top}, follow up to {app.cfg.max_grow} paragraphs")
    table.add_row("context", f"{app.cfg.context_size} tokens")
    table.add_row("verbose", "on" if app.verbose else "off")
    app.console.print(table)


def _toggle(arg: str, current: bool) -> bool:
    return arg.lower() in ("on", "true", "1", "yes") if arg else not current


def handle_slash(app: App, text: str) -> bool:
    """Returns True if the REPL should exit."""
    console, cfg = app.console, app.cfg
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return True
    if cmd == "/help":
        console.print(HELP)
    elif cmd == "/status":
        print_status(app)
    elif cmd == "/sources":
        print_sources(app)
    elif cmd == "/reindex":
        app.index.close()
        app.index = open_index(console, app.index.folder)
    elif cmd == "/folder":
        if not arg:
            console.print(f"folder: {app.index.folder}")
            return False
        folder = Path(arg).expanduser().resolve()
        if not folder.is_dir():
            console.print(f"[red]not a folder: {escape(str(folder))}[/red]")
            return False
        app.index.close()
        app.index = open_index(console, folder)
        app.last = None
        cfg.folder = str(folder)
        cfg.save()
    elif cmd == "/model":
        if not app.available:
            console.print("[red]no model server reachable[/red]")
            return False
        chosen = arg or ui.select(console, "Select the model", cfg.order_models(app.available), current=app.client.model)
        if not chosen:
            console.print("[dim](unchanged)[/dim]")
            return False
        app.client.model = chosen
        app.use_llm = True
        cfg.model = chosen
        cfg.remember_model(chosen)
        cfg.save()
        console.print(f"model: [bold]{escape(chosen)}[/bold]")
    elif cmd in ("/top", "/grow"):
        if not arg.isdigit() or int(arg) < (1 if cmd == "/top" else 0):
            console.print(f"[red]usage: {cmd} <n>[/red]")
            return False
        if cmd == "/top":
            cfg.top = int(arg)
        else:
            cfg.max_grow = int(arg)
        cfg.save()
        console.print(f"{cmd[1:]}: {arg}")
    elif cmd == "/llm":
        want = _toggle(arg, app.use_llm)
        if want and not app.client.model:
            console.print("[red]no model available - is the server running?[/red]")
            return False
        app.use_llm = want
        console.print(f"llm: {'on' if app.use_llm else 'off (search + extracts only)'}")
    elif cmd == "/verbose":
        app.verbose = _toggle(arg, app.verbose)
        cfg.verbose = app.verbose
        cfg.save()
        console.print(f"verbose: {'on' if app.verbose else 'off'}")
    else:
        console.print(f"[red]unknown command: {escape(cmd)}[/red] (try /help)")
    return False


def connect(console: Console, cfg: Config, client: ChatClient, explicit_url: bool) -> list[str]:
    """Chat models on the configured server. If it isn't answering and the URL
    wasn't given on the command line, look for another local server and switch
    to it - starting Ollama instead of Jan shouldn't mean editing a config."""
    models = servers.probe(cfg.base_url, cfg.api_key, timeout=5)
    if models or explicit_url:
        if models is None:
            console.print(f"[yellow]could not reach the model server at {escape(cfg.base_url)}[/yellow]")
        return models or []
    found = [s for s in servers.find_servers(api_key=cfg.api_key) if s.models]
    if not found:
        if models is None:
            console.print(f"[yellow]no model server found (tried {escape(cfg.base_url)} and the usual ports)[/yellow]")
        else:
            console.print(f"[yellow]the server at {escape(cfg.base_url)} has no chat model loaded[/yellow]")
        return []
    server = found[0]
    console.print(f"[dim]found {server.name} at {escape(server.base_url)}[/dim]")
    cfg.base_url = client.base_url = server.base_url
    return server.models


def setup(console: Console, cfg: Config) -> int:
    """Interactive first-run: find (or download) a model, pick a folder, test it."""
    console.print("[bold]localsearch setup[/bold]\n")

    with ui.status(console, "looking for model servers"):
        found = servers.find_servers(extra=(cfg.base_url,), api_key=cfg.api_key)
    for s in found:
        n = len(s.models)
        console.print(f"  [green]✓[/green] {s.name:<10} {escape(s.base_url)}  [dim]{n} model{'s' if n != 1 else ''}[/dim]")
    if not found:
        console.print("  [yellow]no model server is running.[/yellow]")

    with_models = [s for s in found if s.models]
    ollama = next((s for s in found if s.name == "Ollama"), None)

    if not with_models and ollama:
        model = servers.RECOMMENDED_OLLAMA_MODEL
        console.print(f"\nOllama is running but has no models. {model} (~2.5 GB) works well with localsearch.")
        if ui.confirm(console, f"download {model} now?", ui.YES_NO_STOP[:2]) == "yes":
            if not servers.ollama_pull(model):
                console.print("[red]download failed[/red] - try `ollama pull` yourself, then rerun setup.")
                return 1
            ollama.models = servers.probe(ollama.base_url, cfg.api_key) or []
            with_models = [ollama] if ollama.models else []

    if not with_models:
        console.print()
        if servers.ollama_installed() and not ollama:
            console.print("Ollama is installed but not running. Open the Ollama app (or run `ollama serve`),")
            console.print("then run `localsearch setup` again.")
        else:
            console.print("localsearch needs a local model server. The easiest options:")
            console.print("  [bold]Ollama[/bold]     https://ollama.com/download   (then run `localsearch setup` again)")
            console.print("  [bold]Jan[/bold]        https://jan.ai                (start its local API server)")
            console.print("  [bold]LM Studio[/bold]  https://lmstudio.ai           (start its local server)")
        console.print("\n[dim]meanwhile `localsearch --no-llm` searches without a model.[/dim]")
        return 1

    console.print()
    if len(with_models) > 1:
        labels = [f"{s.name}  {s.base_url}" for s in with_models]
        chosen = ui.select(console, "Which server?", labels)
        if not chosen:
            return 1
        server = with_models[labels.index(chosen)]
    else:
        server = with_models[0]

    suggested = servers.pick_default_model(server.models)
    current = cfg.model if cfg.model in server.models else suggested
    options = cfg.order_models(server.models)
    console.print(f"[dim]small (1-4B) models answer fastest here; suggested: {escape(suggested or '')}[/dim]")
    model = ui.select(console, "Which model?", options, current=current)
    if not model:
        return 1

    default_folder = cfg.folder or str(Path.cwd())
    while True:
        raw = ui.ask_line(f"folder to search [{default_folder}]: ", color="1").strip()
        folder = Path(raw or default_folder).expanduser().resolve()
        if folder.is_dir():
            break
        console.print(f"[red]not a folder: {escape(str(folder))}[/red]")

    client = ChatClient(base_url=server.base_url, model=model, api_key=cfg.api_key, no_think=cfg.no_think)
    try:
        with ui.status(console, f"testing {model}"):
            reply = client.ask("Reply with the single word OK.", max_tokens=16)
        console.print(f"[green]✓[/green] {escape(model)} answered: {escape(reply[:40] or '(empty)')}")
    except APIConnectionError as exc:
        console.print(f"[yellow]the model didn't answer: {escape(str(exc))}[/yellow]")
        console.print("[dim]saving anyway - check the model is loaded on the server.[/dim]")

    cfg.base_url = server.base_url
    cfg.model = model
    cfg.remember_model(model)
    cfg.folder = str(folder)
    cfg.save()
    console.print(f"\n[bold]ready.[/bold] run `localsearch` and ask a question about {escape(str(folder))}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    console = Console()
    if argv[:1] == ["setup"]:
        try:
            return setup(console, Config.load())
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]setup cancelled.[/dim]")
            return 1
    args = build_arg_parser().parse_args(argv)
    cfg = Config.load()

    folder = Path(args.folder or cfg.folder or ".").expanduser().resolve()
    if not folder.is_dir():
        console.print(f"[bold red]folder does not exist: {folder}[/bold red]")
        return 1
    if args.top:
        cfg.top = args.top
    if args.max_grow is not None:
        cfg.max_grow = args.max_grow
    if args.context_size:
        cfg.context_size = args.context_size
    if args.base_url:
        cfg.base_url = args.base_url

    client = ChatClient(base_url=cfg.base_url, model="", api_key=cfg.api_key, no_think=not args.think and cfg.no_think)
    available: list[str] = []
    use_llm = not args.no_llm
    if use_llm:
        available = connect(console, cfg, client, explicit_url=bool(args.base_url))
        if not available:
            console.print(
                "[yellow]no model available - running search + extracts only (like --no-llm).[/yellow]\n"
                "[dim]run `localsearch setup` to find or download a model.[/dim]"
            )
            use_llm = False

    if available:
        model = args.model or (cfg.model if cfg.model in available else None) or servers.pick_default_model(available)
        if args.model and args.model not in available:
            console.print(f"[yellow]warning: '{escape(args.model)}' isn't loaded on the server[/yellow]")
        client.model = model
        cfg.model = model
        cfg.remember_model(model)

    cfg.folder = str(folder)
    cfg.save()

    console.print(f"[bold]localsearch[/bold]  {folder}")
    index = open_index(console, folder)
    app = App(console, cfg, client, index, available, use_llm, args.verbose or cfg.verbose)

    if args.question:
        answer(app, " ".join(args.question))
        index.close()
        return 0

    print_status(app)
    console.print("[dim]ask a question, /help for commands, /model to switch models.[/dim]")
    console.print()
    completion.install(get_models=lambda: app.available)

    while True:
        try:
            text = ui.ask_line("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not text:
            continue
        if text.startswith("/"):
            if handle_slash(app, text):
                break
            continue
        answer(app, text)

    app.index.close()
    console.print("bye.")
    return 0
