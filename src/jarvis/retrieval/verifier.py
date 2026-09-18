"""LLM-based verification for intent matches (shared by skills and MCP).

After the retriever + gate selects candidates, a lightweight LLM call decides
whether the user's request *actually* needs each candidate. It returns one of
three states:

  * ``confirmed`` — the LLM picked specific candidates; inject them.
  * ``none``      — the LLM says nothing applies; inject nothing.
  * ``unclear``   — the request is too vague to tell (or the LLM was
                    unreachable / unparseable). Don't act; let the caller ask
                    the user to clarify instead of guessing.

The ``unclear`` state is deliberate: an ambiguous prompt like "更新一下" must
not silently trigger ``update-deps``. When we can't confirm intent we surface
the candidates for clarification rather than command-injecting them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
from loguru import logger

from ..config import OllamaConfig
from .retriever import Match

CONFIRMED = "confirmed"
NONE = "none"
UNCLEAR = "unclear"


@dataclass
class VerifyResult:
    """Outcome of intent verification. ``matches`` is non-empty only when
    ``status == CONFIRMED``."""

    status: str
    matches: list[Match] = field(default_factory=list)


def _system_prompt(noun: str) -> str:
    return (
        f"You decide which {noun}s a user request actually needs.\n"
        f"- If one or more are clearly needed, reply with their numbers "
        "(comma-separated).\n"
        f"- If none apply, reply \"none\".\n"
        "- If the request is too vague to tell which (if any) is needed, "
        "reply \"unclear\".\n"
        "Reply with ONLY one of those. No explanation."
    )


async def verify_candidates(
    text: str,
    candidates: list[Match],
    ollama_cfg: OllamaConfig,
    *,
    noun: str = "tool server",
    timeout_s: float = 1.5,
) -> VerifyResult:
    """Classify *candidates* against the request via a local LLM.

    *noun* names the candidate kind in the prompt ("skill", "tool server") so
    the model judges in the right frame.

    Returns a :class:`VerifyResult`. On any failure (LLM down, timeout,
    unparseable reply) the status is ``UNCLEAR`` — we never fabricate a
    ``confirmed`` we couldn't actually confirm, so a flaky Ollama degrades to
    "ask the user", not "command-inject everything the gate let through".
    """
    if not candidates:
        return VerifyResult(NONE)

    cand_lines = "\n".join(
        f"{i + 1}. {m.record.name} — {m.record.description}"
        for i, m in enumerate(candidates)
    )
    messages = [
        {"role": "system", "content": _system_prompt(noun)},
        {
            "role": "user",
            "content": f"Request: \"{text}\"\n\n{noun.title()}s:\n{cand_lines}",
        },
    ]

    try:
        # trust_env=False: skip the macOS system proxy for the local server
        # (see phrase/providers/ollama.py for why it breaks the MLX shim).
        async with httpx.AsyncClient(
            base_url=ollama_cfg.base_url, timeout=timeout_s, trust_env=False,
        ) as client:
            r = await client.post(
                "/api/chat",
                json={
                    "model": ollama_cfg.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    # Keep the model resident between bursty requests so the
                    # next verification doesn't pay a cold reload (see OllamaConfig).
                    "keep_alive": ollama_cfg.keep_alive,
                    "options": {"temperature": 0, "num_predict": 20},
                },
            )
            r.raise_for_status()
            reply = r.json()["message"]["content"].strip().lower()
    except Exception as exc:  # noqa: BLE001 — unreachable LLM => unclear, not guess
        logger.debug("verifier unavailable ({}); intent unclear", exc)
        return VerifyResult(UNCLEAR)

    logger.debug("verifier reply={!r} for {!r}", reply, text[:60])

    # "unclear" wins over "none" if the model hedged with both.
    if UNCLEAR in reply:
        return VerifyResult(UNCLEAR)
    if NONE in reply:
        return VerifyResult(NONE)

    indices: set[int] = set()
    for tok in re.findall(r"\d+", reply):
        idx = int(tok) - 1
        if 0 <= idx < len(candidates):
            indices.add(idx)

    # Numbers present -> confirmed subset (in candidate order, i.e. by score).
    # Nothing parseable and no "none" keyword -> we couldn't read the verdict,
    # so treat it as unclear rather than passing every candidate through.
    if not indices:
        return VerifyResult(UNCLEAR)

    return VerifyResult(CONFIRMED, [candidates[i] for i in sorted(indices)])
