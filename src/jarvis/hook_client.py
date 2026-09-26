"""Thin client invoked by Claude Code Notification hook.

Reads JSON payload from stdin, writes a single NDJSON line over the
configured Unix socket, and exits. Must never raise to stdout — Claude
Code reads stdout for hook decisions.
"""
from __future__ import annotations

import io
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import IO

from .config import DEFAULT_CONFIG_PATH, load_config

# CC's PreToolUse(AskUserQuestion) payload has no `notification_type` field
# — the listener would drop it. We translate it here into a daemon event
# whose `text` is pre-baked, so synthesis skips the LLM (the daemon already
# special-cases `text=...` for the manual `jarvis say --text` path).
#
# Output language is governed by `behavior.voice_language` ("en" | "zh" |
# "auto"). Default is "en" — the user's chosen British voice identity.
# "auto" picks per-event from the question text (CJK → zh).
_EN_ORDINALS = ("Option one", "Option two", "Option three", "Option four")
_ZH_ORDINALS = ("选项一", "选项二", "选项三", "选项四")


_CODEX_SESSIONS_DIR = Path(os.path.expanduser("~/.jarvis/.codex-sessions"))


def _is_first_codex_turn(thread_id: str) -> bool:
    """File-based first-seen check for Codex thread-ids."""
    try:
        _CODEX_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        marker = _CODEX_SESSIONS_DIR / thread_id
        if marker.exists():
            return False
        marker.write_text("", encoding="utf-8")
        return True
    except OSError:
        return False


def _is_muted() -> bool:
    """Per-session mute switch. Spawned sub-Claude sessions (eg orchestrate's
    spawn-agent.sh) launch as `JARVIS_MUTE=1 claude …`; hooks inherit the env,
    so the whole session stays silent while user-opened sessions are
    unaffected. SessionStart payloads carry no field that distinguishes a
    programmatic spawn from a user launch, hence the env-var channel."""
    value = os.environ.get("JARVIS_MUTE", "")
    return value.lower() not in ("", "0", "false")


def _hook_debug(template: str, *args: object) -> None:
    """Diagnostic line on stderr, only when `JARVIS_HOOK_DEBUG` is set.

    The hook runs as a short-lived subprocess with no logger; silent drops are
    otherwise invisible, and this project has lost days to audio that stopped
    with no trace. Off by default so CC never shows hook noise to the user.
    """
    if os.environ.get("JARVIS_HOOK_DEBUG", "").lower() in ("", "0", "false"):
        return
    try:
        sys.stderr.write("jarvis-hook: " + template.format(*args) + "\n")
    except Exception:  # noqa: BLE001 — diagnostics must never break the hook
        pass


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _questions_have_cjk(questions: list) -> bool:
    """CJK anywhere in the first question's text or option labels. Strict on
    purpose: an English question with Chinese labels must not verbatim-render
    into the English voice — it goes to the LLM for translation instead."""
    first = questions[0]
    if _has_cjk(first.get("question") or ""):
        return True
    for opt in first.get("options") or []:
        if isinstance(opt, dict) and _has_cjk(opt.get("label") or ""):
            return True
    return False


def _resolve_lang(mode: str, questions: list) -> str:
    """Pick output lang from the user's mode setting and question content."""
    if mode == "zh":
        return "zh"
    if mode == "auto":
        first_q = ""
        if questions and isinstance(questions[0], dict):
            first_q = questions[0].get("question") or ""
        return "zh" if _has_cjk(first_q) else "en"
    return "en"  # "en" and any unrecognized value


def _render_askuserquestion(questions: list, lang: str) -> str | None:
    if not questions or not isinstance(questions[0], dict):
        return None
    first = questions[0]
    q_text = (first.get("question") or "").strip()
    options = first.get("options") or []
    if not q_text or not options:
        return None

    q_clean = q_text.rstrip("。.!?！？")
    if lang == "zh":
        ordinals = _ZH_ORDINALS
        head = q_clean if q_clean.startswith(("先生", "Sir,", "Sir ")) else "先生，" + q_clean
        sep = "。"
        remaining_tmpl = "另有 {n} 个问题等候，先生。"
    else:
        ordinals = _EN_ORDINALS
        head = (
            q_clean
            if q_clean.lower().startswith(("sir,", "sir "))
            else "Sir, " + q_clean
        )
        sep = ". "
        remaining_tmpl = "And {n} more questions on screen, sir."

    parts = [head + sep]
    for i, opt in enumerate(options[:4]):
        if not isinstance(opt, dict):
            continue
        label = (opt.get("label") or "").strip()
        if not label:
            continue
        parts.append(f"{ordinals[i]}: {label}{sep}")

    remaining = len(questions) - 1
    if remaining > 0:
        parts.append(remaining_tmpl.format(n=remaining))
    return "".join(parts).rstrip()


def _first_question_is_usable(questions: list) -> bool:
    if not questions or not isinstance(questions[0], dict):
        return False
    first = questions[0]
    q_text = (first.get("question") or "").strip()
    options = first.get("options") or []
    return bool(q_text and options)


def _translate_cc_payload(payload: dict, lang_mode: str = "en") -> dict | None:
    """Rewrite a raw hook payload (Claude Code OR Codex CLI) into the
    daemon's normalized shape.

    Both products send lifecycle-hook payloads with the same snake_case
    shape and overlapping `hook_event_name` vocabulary (UserPromptSubmit,
    PostToolUse, PreToolUse), so those code paths are shared. Codex-only
    pieces handled here:
      * `PermissionRequest` → translated to a permission_prompt event so
        the daemon's existing routing pipeline can speak it.
      * `notify`'s flat `agent-turn-complete` payload (kebab-case, no
        `hook_event_name`) → translated to idle_prompt.

    Returns the new dict, or None if the payload can't/shouldn't be
    forwarded. Returns the original (unchanged) for payloads already in
    daemon shape.

    For AskUserQuestion the behavior depends on `lang_mode`:
      - "auto" → render verbatim text + lang in the hook (no LLM round-trip);
        CJK question text picks zh, otherwise en.
      - "en" / "zh" → forward the questions WITHOUT text/lang so the daemon's
        phrase router calls Ollama/DeepSeek to translate-and-rephrase into a
        Jarvis-toned line in the user's chosen output language.
    """
    # Codex `notify` payload — flat, kebab-case keys, `type` discriminator.
    # Currently only `agent-turn-complete` is emitted, but check by type
    # rather than presence so future Codex notify types fall through to
    # a no-op rather than triggering the wrong event.
    if payload.get("type") == "agent-turn-complete":
        return {
            "notification_type": "idle_prompt",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("thread-id"),
        }

    hook_event = payload.get("hook_event_name")
    if hook_event == "SessionStart":
        # Only the genuine cold start gets the full briefing. CC also sends
        # this event on `/clear`, compaction and resumed sessions; those
        # shouldn't blast the user with a fresh briefing every time (resume
        # gets a one-line welcome instead, see below). `source` is one of
        # "startup", "resume", "clear", "compact" — both clients use the
        # same field.
        source = payload.get("source")
        if source == "resume":
            # A resumed session gets a short LLM-phrased "welcome back"
            # instead of the full briefing — silence here read as "Jarvis
            # is broken" (2026-09-26, `claude --resume` in another tab).
            # session_id dropped for the same first-prompt-cancel reason as
            # the briefing below; the project name gives the line a hook.
            cwd = payload.get("cwd")
            return {
                "notification_type": "session_resume",
                "tool_name": None,
                "tool_input": {"project": Path(cwd).name} if cwd else {},
                "cwd": cwd,
                "session_id": None,
            }
        if source and source != "startup":
            return None
        # Deliberately drop the session_id. SessionStart is immediately
        # followed by the user's first UserPromptSubmit (in `codex exec` the
        # prompt is supplied up front; in the interactive TUI the user types
        # within the briefing's 10-40s Ollama compose window), which carries
        # the SAME session_id and translates to a `cancel`. With the id
        # attached, that cancel drops the still-composing briefing from the
        # daemon queue (or flags it so the worker skips synth) and the user
        # hears nothing. The briefing is a fire-once greeting, not an
        # "awaiting your input" prompt, so it has no business sharing the
        # session's cancel identity — leaving session_id unset makes it
        # immune to the cancel and it always speaks. dedup_key is (cwd, type,
        # tool) so multi-tab dedup is unaffected.
        return {
            "notification_type": "session_start",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": None,
        }
    # A tool/command failed. CC fires `PostToolUseFailure` carrying the
    # tool name + a `tool_response` (error gist). We forward the raw
    # tool_input/tool_response so the daemon's extract() can summarize it;
    # phrasing (grave tone) happens in the daemon, not here.
    if hook_event == "PostToolUseFailure":
        ti = payload.get("tool_input") or {}
        resp = payload.get("tool_response")
        if isinstance(resp, dict) or isinstance(resp, str):
            ti = {**ti, "tool_response": resp}
        return {
            "notification_type": "tool_failure",
            "tool_name": payload.get("tool_name"),
            "tool_input": ti,
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # Claude finished responding. CC fires `Stop` (and `SubagentStop` for a
    # finished sub-agent turn). A brief completion line; dedup keeps it from
    # chattering when several Stops land in one window.
    if hook_event in ("Stop", "SubagentStop"):
        return {
            "notification_type": "task_complete",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    if hook_event in ("UserPromptSubmit", "PostToolUse"):
        sid = payload.get("session_id")
        if not sid:
            return None
        return {"command": "cancel", "session_id": sid}
    # Codex permission prompt — no Claude Code analogue. Translate to
    # the daemon's permission_prompt notification so the existing phrase
    # router + TTS pipeline handles it identically.
    if hook_event == "PermissionRequest":
        return {
            "notification_type": "permission_prompt",
            "tool_name": payload.get("tool_name"),
            "tool_input": payload.get("tool_input") or {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # --- Tier 1 lifecycle events ---
    # Context about to be compressed.
    if hook_event == "PreCompact":
        return {
            "notification_type": "context_compacting",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # API rate limit hit — Claude pauses.
    if hook_event == "RateLimitError":
        return {
            "notification_type": "rate_limited",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # A sub-agent was dispatched.
    if hook_event == "SubagentStart":
        return {
            "notification_type": "subagent_spawned",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # Turn limit reached — Claude stopped.
    if hook_event == "MaxTurnsReached":
        return {
            "notification_type": "max_turns_reached",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # --- Tier 2 lifecycle events ---
    # API returned an error (not rate-limit — a general failure).
    if hook_event == "APIError":
        ti = {}
        # Stash the error info so the phrase router can extract a gist.
        for key in ("error", "message", "status_code"):
            val = payload.get(key)
            if val is not None:
                ti[key] = val
        return {
            "notification_type": "api_error",
            "tool_name": None,
            "tool_input": ti,
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # Session ended.
    if hook_event == "SessionStop":
        return {
            "notification_type": "session_end",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # Context compression finished.
    if hook_event == "PostCompact":
        return {
            "notification_type": "context_compacted",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    # Context window full.
    if hook_event == "ContextWindowOverflow":
        return {
            "notification_type": "context_overflow",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    if hook_event == "PreToolUse" and \
            payload.get("tool_name") == "AskUserQuestion":
        ti = payload.get("tool_input") or {}
        questions = ti.get("questions")
        if not isinstance(questions, list) or not _first_question_is_usable(questions):
            return None

        base: dict = {
            "notification_type": "ask_user_question",
            "tool_name": "AskUserQuestion",
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }

        if lang_mode == "auto":
            lang = _resolve_lang("auto", questions)
            text = _render_askuserquestion(questions, lang)
            if not text:
                return None
            return {**base, "tool_input": {}, "text": text, "lang": lang}

        # "en" / "zh": when the question is ALREADY in the target language,
        # render verbatim here — no LLM round-trip. The LLM leg exists to
        # translate; same-language "rephrasing" only adds its full latency
        # (first audio 16-60s late in practice) and lets a small model merge
        # or drop options. Only cross-language questions still go through
        # the daemon's phrase router.
        if lang_mode == "en" and not _questions_have_cjk(questions):
            text = _render_askuserquestion(questions, "en")
            if text:
                return {**base, "tool_input": {}, "text": text, "lang": "en"}
        elif lang_mode == "zh" and _has_cjk(
            (questions[0].get("question") or "")
        ):
            text = _render_askuserquestion(questions, "zh")
            if text:
                return {**base, "tool_input": {}, "text": text, "lang": "zh"}

        # Cross-language (or render failure): forward verbatim questions so
        # the daemon's phrase router can translate them into the configured
        # voice_language.
        return {**base, "tool_input": {"questions": questions}}
    return payload


def forward_event(
    stream: IO[str],
    sock_path: str | Path,
    *,
    lang_mode: str = "en",
    cancel_on_user_action: bool = True,
    mute_subagent_events: bool = True,
) -> bool:
    """Forward an NDJSON event from `stream` to the unix socket at `sock_path`.

    `lang_mode` ("en" | "zh" | "auto") only affects AskUserQuestion translation.
    `cancel_on_user_action`: when False, UserPromptSubmit / PostToolUse hook
    payloads are dropped without contacting the daemon.
    `mute_subagent_events`: when True, payloads fired from inside a subagent's
    work (CC stamps those with both `agent_id` and `agent_type`) are dropped —
    the subagent's tool calls, failures, and prompts stay silent while
    main-session events speak. SubagentStart carries the same pair but is
    exempt: it is a session-level lifecycle notice with its own notification
    type, governed by the events allowlist instead. SubagentStop is *not*
    exempt — it translates to the same `task_complete` as the main session's
    Stop, so exempting it would announce "all done" once per finished subagent.
    When the `JARVIS_MUTE` env var is set (anything but ""/"0"/"false"), every
    event is dropped before the stream is read — the session is fully silent.

    Returns True if successfully sent. Returns False on any failure
    (invalid JSON, socket missing, write error, dropped by policy) —
    never raises.
    """
    if _is_muted():
        return False

    sock_path = Path(sock_path)
    try:
        raw = stream.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False

    if not isinstance(payload, dict):
        return False

    if (
        mute_subagent_events
        # Both fields, not just `agent_id`: two corroborating markers make a
        # silent global mute far less likely if a future CC build starts
        # stamping one of them on main-session payloads too. Losing the mute
        # (chatter returns) is a recoverable failure; losing every event is
        # the multi-day silent-audio failure this project has hit before.
        and payload.get("agent_id")
        and payload.get("agent_type")
        and payload.get("hook_event_name") != "SubagentStart"
    ):
        _hook_debug(
            "dropped {} from subagent {} ({})",
            payload.get("hook_event_name"),
            payload.get("agent_id"),
            payload.get("agent_type"),
        )
        return False

    if (
        not cancel_on_user_action
        and payload.get("hook_event_name") in ("UserPromptSubmit", "PostToolUse")
    ):
        return False

    # Codex CLI v0.141+ does not fire SessionStart hooks. On the first
    # agent-turn-complete for each thread, synthesize a session_start so
    # the daemon's briefing pipeline triggers.
    synth_start: dict | None = None
    if (
        payload.get("type") == "agent-turn-complete"
        and (tid := payload.get("thread-id"))
        and _is_first_codex_turn(tid)
    ):
        synth_start = {
            "notification_type": "session_start",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": None,
            "_received_at": time.time(),
        }

    payload = _translate_cc_payload(payload, lang_mode=lang_mode)
    if payload is None and synth_start is None:
        return False

    def _send(data: dict) -> bool:
        data.setdefault("_received_at", time.time())
        line = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(0.5)
            s.connect(str(sock_path))
            s.sendall(line)
            return True
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            return False
        finally:
            try:
                s.close()
            except OSError:
                pass

    ok = False
    if synth_start is not None:
        ok = _send(synth_start) or ok
    if payload is not None:
        ok = _send(payload) or ok
    return ok


def _request_reply(sock_path: str | Path, payload: dict, timeout_s: float) -> dict | None:
    """Send one JSON line and read one JSON line back. Returns None on any
    failure (daemon down, timeout, bad reply) — the caller injects nothing."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(timeout_s)
        s.connect(str(sock_path))
        s.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b"\n", 1)[0].decode("utf-8", errors="replace")
        return json.loads(line) if line.strip() else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    finally:
        try:
            s.close()
        except OSError:
            pass


def _prompt_text(payload: dict) -> str:
    """Extract the user's prompt from a UserPromptSubmit payload (CC uses
    `prompt`; tolerate Codex/other spellings)."""
    for key in ("prompt", "user_prompt", "current_prompt", "text"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _emit_additional_context(context: str) -> None:
    """Print the CC/Codex UserPromptSubmit injection envelope. This is the one
    place the hook writes stdout — only with a valid additionalContext payload,
    always exit 0 so the prompt is never blocked."""
    sys.stdout.write(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            },
            ensure_ascii=False,
        )
    )
    sys.stdout.flush()


def maybe_inject_skills(raw: str, cfg) -> None:
    """For UserPromptSubmit, ask the daemon for relevant skills and inject them;
    for SessionStart, kick a best-effort index refresh. No-op unless skills are
    enabled. Never raises, never blocks the prompt beyond the configured budget."""
    if not getattr(cfg.skills, "enabled", False):
        return
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(payload, dict):
        return
    event = payload.get("hook_event_name")
    timeout_s = max(0.05, cfg.skills.query_timeout_ms / 1000.0)

    if event == "SessionStart":
        # Refresh the index on cold start so newly installed skills get picked
        # up. Fire-and-forget: short timeout, reply ignored.
        _request_reply(cfg.paths.socket, {"command": "skill_refresh"}, timeout_s)
        return

    if event != "UserPromptSubmit":
        return
    text = _prompt_text(payload)
    if not text.strip():
        return
    reply = _request_reply(
        cfg.paths.socket,
        {
            "command": "skill_query",
            "text": text,
            "session_id": payload.get("session_id"),
        },
        timeout_s,
    )
    if reply and isinstance(reply, dict) and reply.get("context"):
        _emit_additional_context(str(reply["context"]))


def main() -> int:
    """Entry point registered as `jarvis-hook` console_script.

    Must NEVER raise — Claude Code reads stdout for hook decisions and a
    traceback would corrupt that channel. All failures are silent and
    exit 0. The only stdout this writes is a UserPromptSubmit additionalContext
    envelope when a skill matches (see `maybe_inject_skills`).
    """
    try:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        mode = getattr(cfg.behavior, "voice_language", "en") or "en"
        cancel = getattr(cfg.behavior, "cancel_on_user_action", True)
        mute_sub = getattr(cfg.behavior, "mute_subagent_events", True)
        # Read stdin once; feed both the (fire-and-forget) TTS path and the
        # (request/response) skills path. TTS first so a cancel reaches the
        # daemon before we spend the skill-query round-trip.
        raw = sys.stdin.read()
        forward_event(
            io.StringIO(raw),
            cfg.paths.socket,
            lang_mode=mode,
            cancel_on_user_action=cancel,
            mute_subagent_events=mute_sub,
        )
        maybe_inject_skills(raw, cfg)
    except Exception:  # noqa: BLE001 — structural guarantee
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
