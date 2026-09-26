"""Shared dataclasses used across hook_client, daemon, phrase, tts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NotificationType = Literal[
    "permission_prompt",
    "idle_prompt",
    "elicitation_dialog",
    "ask_user_question",
    # New CC/Codex session — daemon composes a Jarvis briefing
    # (greeting + local time + weather) instead of going via the LLM router.
    "session_start",
    # A session was resumed (`claude --resume`) — a short LLM-phrased
    # "welcome back", not the full briefing.
    "session_resume",
    # A tool/command failed (CC PostToolUseFailure). Jarvis speaks a short,
    # graver-toned line naming the failed tool + the error gist.
    "tool_failure",
    # Claude finished responding (CC Stop). Jarvis gives a brief completion
    # line. Fires often, so leans on the dedup window to stay quiet.
    "task_complete",
    # --- Tier 1: high-value lifecycle events ---
    # Context is about to be compressed (CC PreCompact).
    "context_compacting",
    # Hit the API rate limit (CC RateLimitError).
    "rate_limited",
    # A sub-agent was dispatched (CC SubagentStart).
    "subagent_spawned",
    # Turn limit reached — Claude stopped (CC MaxTurnsReached).
    "max_turns_reached",
    # --- Tier 2: medium-value lifecycle events ---
    # The API returned an error (CC APIError).
    "api_error",
    # Session ended (CC SessionStop).
    "session_end",
    # Context compression finished (CC PostCompact).
    "context_compacted",
    # Context window is full (CC ContextWindowOverflow).
    "context_overflow",
]

Lang = Literal["zh", "en"]

# ---------------------------------------------------------------------------
# Emotional tone — threaded from event type → phrase prompt → TTS parameters.
# Providers that support voice_settings (ElevenLabs) map these to synthesis
# knobs; all providers benefit from the prompt clause, which shapes the LLM's
# written text toward the target emotion.
# ---------------------------------------------------------------------------

Emotion = Literal["warm", "neutral", "gentle", "grave", "pleased", "sardonic"]

EVENT_EMOTION: dict[str, Emotion] = {
    "session_start": "warm",
    "session_resume": "warm",
    "permission_prompt": "neutral",
    "idle_prompt": "gentle",
    "tool_failure": "grave",
    "task_complete": "pleased",
    "ask_user_question": "neutral",
    "elicitation_dialog": "neutral",
}


def emotion_for(notification_type: str) -> Emotion:
    """Look up the emotion for an event type; unknown types default to neutral."""
    return EVENT_EMOTION.get(notification_type, "neutral")


@dataclass(frozen=True)
class Event:
    """A single notification event from Claude Code, normalized for daemon use."""

    notification_type: NotificationType
    tool_name: str | None
    tool_input: dict[str, Any] = field(default_factory=dict)
    cwd: str | None = None
    session_id: str | None = None
    raw_message: str | None = None
    received_at: float = 0.0  # epoch seconds; filled by listener
    # Pre-baked text from the caller. When set the daemon SKIPS the phrase
    # router (no LLM round-trip) and synthesizes this string verbatim. Used
    # by `jarvis say --text` for assistant-side scenarios CC doesn't cover.
    text: str | None = None
    lang: Lang | None = None  # only honored when `text` is set
    voice_id: str | None = None  # per-event TTS voice override (eg EL voice_id)

    def dedup_key(self) -> str:
        """Hash key for dedup window: same (cwd, type, tool) collapses."""
        return f"{self.cwd or ''}::{self.notification_type}::{self.tool_name or ''}"
