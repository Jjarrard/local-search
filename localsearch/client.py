"""Minimal OpenAI-compatible chat client (Jan, Ollama, LM Studio, llama.cpp...)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import requests


class APIConnectionError(RuntimeError):
    pass


# Model ids that are embedding models, not chat models - never pick one by default.
NON_CHAT_HINTS = ("embed", "sentence-transformer", "bge-", "e5-", "rerank")


def is_chat_model(name: str) -> bool:
    return not any(h in name.lower() for h in NON_CHAT_HINTS)


THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop <think>...</think> (or an unclosed <think> to end of text)."""
    return THINK_BLOCK.sub("", text).strip()


class ThinkFilter:
    """Streaming version of strip_think: feed deltas, get back visible text.

    Reasoning models sometimes think even when asked not to. Their thoughts
    are never the answer, so they are never shown.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._thinking = False
        self._decided = False

    def feed(self, piece: str) -> str:
        self._buf += piece
        out = ""
        while self._buf:
            if self._thinking:
                end = self._buf.find("</think>")
                if end == -1:
                    self._buf = self._buf[-len("</think>"):]  # keep a possible partial tag
                    return out
                self._buf = self._buf[end + len("</think>"):]
                self._thinking = False
                continue
            if not self._decided:
                stripped = self._buf.lstrip()
                if len(stripped) < len("<think>") and "<think>".startswith(stripped):
                    return out  # might still become a <think> tag
                self._decided = True
                if stripped.startswith("<think>"):
                    self._buf = stripped[len("<think>"):]
                    self._thinking = True
                    continue
                self._buf = stripped
            out += self._buf
            self._buf = ""
        return out


@dataclass
class ChatClient:
    base_url: str = "http://127.0.0.1:1338/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    temperature: float = 0.0
    timeout: float = 120.0
    no_think: bool = True

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def list_models(self) -> list[str]:
        url = f"{self.base_url.rstrip('/')}/models"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise APIConnectionError(f"could not reach the model server at {url}: {exc}") from exc
        data = resp.json()
        return [m.get("id", "?") for m in data.get("data", [])]

    def stream_chat(self, messages: list[dict], max_tokens: int = 256) -> Iterator[str]:
        """Yield raw text deltas from a streamed chat completion."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if self.no_think:
            # llama.cpp (Jan) honours this for Qwen3-style templates; other
            # servers ignore unknown fields.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = requests.post(url, headers=self._headers(), json=payload, stream=True, timeout=self.timeout)
        except requests.RequestException as exc:
            raise APIConnectionError(
                f"could not reach the model server at {url} ({exc}). Is it running with a model loaded?"
            ) from exc
        if not resp.ok:
            body = resp.text.strip()
            raise APIConnectionError(f"server rejected the request ({resp.status_code}): {body[:500] or '(no body)'}")
        # SSE responses often omit a charset, and requests then assumes
        # latin-1 - which turns every curly quote into "â€™" garbage.
        resp.encoding = "utf-8"

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                payload_str = raw_line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    yield piece
        finally:
            resp.close()

    def stream_answer(self, prompt: str, max_tokens: int = 256) -> Iterator[str]:
        """Stream visible text for a single-prompt question, thoughts removed."""
        filt = ThinkFilter()
        for piece in self.stream_chat([{"role": "user", "content": prompt}], max_tokens=max_tokens):
            visible = filt.feed(piece)
            if visible:
                yield visible

    def ask(self, prompt: str, max_tokens: int = 256) -> str:
        """One prompt in, one cleaned-up answer out. The only call the pipeline needs."""
        raw = "".join(self.stream_chat([{"role": "user", "content": prompt}], max_tokens=max_tokens))
        return strip_think(raw)
