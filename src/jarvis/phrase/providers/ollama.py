"""Ollama local LLM provider (uses /api/chat).

Also fronts the MLX shim (``~/.vimo/mlx-ollama``), which speaks the same
``/api/chat`` NDJSON on the same port since 2026-09-18.

Every client here is built with ``trust_env=False``: httpx otherwise picks up
the macOS *system* proxy (Clash on 127.0.0.1:7897) and sends the request to
it in absolute-form (``GET http://localhost:11434/api/chat``). Real Ollama's
Go server tolerated that; the shim's uvicorn does not strip the scheme/host
and answers 404 for every route. The system exception list nominally covers
``localhost`` but is stored as one space-joined string, so Python's bypass
check never matches it. A local server never needs a proxy anyway.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...config import OllamaConfig
from .base import PhraseProvider


class OllamaProvider(PhraseProvider):
    name = "ollama"

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    def _payload(self, messages: list[dict[str, str]], *, stream: bool) -> dict[str, Any]:
        """Request body for `/api/chat`, shared by both call paths."""
        return {
            "model": self.cfg.model,
            "messages": messages,
            "stream": stream,
            # think=False disables Qwen3/DeepSeek-R1 style chain-of-thought
            # output so num_predict isn't consumed by <think>...</think>.
            # Ignored by models that don't emit thinking tokens.
            "think": False,
            # Keep the model resident between bursty notifications so the
            # next event doesn't pay a cold reload (see OllamaConfig).
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": 0.7,
                "num_predict": 200,
                # Explicit context size so the KV cache is sized for our
                # ~1k-token prompts instead of the server default (32k).
                "num_ctx": self.cfg.num_ctx,
            },
        }

    async def generate(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds,
            trust_env=False,
        ) as client:
            r = await client.post(
                "/api/chat",
                json=self._payload(messages, stream=False),
            )
            r.raise_for_status()
            data = r.json()
        return data["message"]["content"].strip()

    async def generate_stream(
        self, messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Stream from Ollama's ``/api/chat`` with ``stream: true``.

        Ollama returns NDJSON: one JSON object per line with
        ``message.content`` holding the token delta.
        """
        async with httpx.AsyncClient(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_seconds,
            trust_env=False,
        ) as client:
            async with client.stream(
                "POST",
                "/api/chat",
                json=self._payload(messages, stream=True),
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(
                base_url=self.cfg.base_url, timeout=1.0, trust_env=False,
            ) as c:
                r = await c.get("/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
