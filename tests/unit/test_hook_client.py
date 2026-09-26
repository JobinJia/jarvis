import io
import json
import socket
import threading
from pathlib import Path

from jarvis.hook_client import forward_event


def _start_unix_echo_server(path: Path, max_conns: int = 1) -> list[bytes]:
    received: list[bytes] = []
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(max_conns)

    def _serve():
        for _ in range(max_conns):
            try:
                conn, _ = sock.accept()
                received.append(conn.recv(4096))
                conn.close()
            except OSError:
                break
        sock.close()

    threading.Thread(target=_serve, daemon=True).start()
    return received


def test_forward_event_writes_ndjson_to_socket(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)

    payload = {
        "session_id": "abc",
        "notification_type": "permission_prompt",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)

    assert ok is True
    # Allow the server thread to write `received`
    for _ in range(50):
        if received:
            break
        import time
        time.sleep(0.01)
    assert len(received) == 1
    line = received[0].decode().strip()
    assert json.loads(line)["tool_name"] == "Bash"


def test_forward_event_returns_false_when_socket_missing(tmp_path: Path):
    sock_path = tmp_path / "does-not-exist.sock"
    ok = forward_event(io.StringIO('{"notification_type":"idle_prompt"}'), sock_path)
    assert ok is False


def test_forward_event_handles_invalid_json(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    ok = forward_event(io.StringIO("not-json"), sock_path)
    assert ok is False


def test_forward_event_returns_false_for_non_dict_json(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    # Valid JSON, but a list — would have crashed on payload["_received_at"]=...
    ok = forward_event(io.StringIO("[1,2,3]"), sock_path)
    assert ok is False


def _recv_one(received: list[bytes]) -> dict:
    import time as _t

    for _ in range(50):
        if received:
            break
        _t.sleep(0.01)
    assert received, "server thread received nothing"
    return json.loads(received[0].decode().strip())


def _recv_n(received: list[bytes], n: int) -> list[dict]:
    import time as _t

    for _ in range(100):
        if len(received) >= n:
            break
        _t.sleep(0.01)
    assert len(received) >= n, f"expected {n} messages, got {len(received)}"
    return [json.loads(b.decode().strip()) for b in received[:n]]


def test_forward_event_userpromptsubmit_sends_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "abc-123",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row["command"] == "cancel"
    assert row["session_id"] == "abc-123"


def test_forward_event_posttooluse_sends_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "sess-9",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row["command"] == "cancel"
    assert row["session_id"] == "sess-9"


def test_forward_event_userpromptsubmit_without_session_id_is_dropped(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {"hook_event_name": "UserPromptSubmit", "cwd": "/x"}
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is False


def test_forward_event_cancel_disabled_by_flag(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "abc",
    }
    ok = forward_event(
        io.StringIO(json.dumps(payload)),
        sock_path,
        cancel_on_user_action=False,
    )
    assert ok is False


# --- subagent-origin muting -------------------------------------------------
# CC stamps every hook payload fired from inside a subagent's work with
# `agent_id` + `agent_type` (verified empirically 2026-07-18 against a live
# headless session); main-session payloads carry neither, and session_id is
# identical for both — so that pair is the only marker separating them. We
# require BOTH fields: one of them turning up on a main-session payload in a
# future CC build would otherwise silence Jarvis globally with no trace.


def test_forward_event_drops_subagent_tool_failure_by_default(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "false"},
        "session_id": "sess-1",
        "agent_id": "ad419dede82c8f0c1",
        "agent_type": "general-purpose",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is False


def test_forward_event_keeps_subagent_events_when_mute_disabled(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "false"},
        "session_id": "sess-1",
        "agent_id": "ad419dede82c8f0c1",
        "agent_type": "general-purpose",
        "cwd": "/x",
    }
    ok = forward_event(
        io.StringIO(json.dumps(payload)), sock_path, mute_subagent_events=False,
    )
    assert ok is True
    row = _recv_one(received)
    assert row["notification_type"] == "tool_failure"


def test_forward_event_subagent_start_exempt_from_mute(tmp_path: Path):
    """SubagentStart carries agent_id too, but it is a session-level lifecycle
    notice with its own notification type — the events allowlist governs it,
    not this mute."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "SubagentStart",
        "session_id": "sess-1",
        "agent_id": "ad419dede82c8f0c1",
        "agent_type": "general-purpose",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row["notification_type"] == "subagent_spawned"


def test_forward_event_drops_subagent_stop(tmp_path: Path):
    """SubagentStop is NOT exempt: it translates to the same `task_complete` as
    the main session's Stop, so exempting it would announce "all done" once per
    finished subagent — exactly the chatter this mute exists to remove."""
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "SubagentStop",
        "session_id": "sess-1",
        "agent_id": "ad419dede82c8f0c1",
        "agent_type": "general-purpose",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is False


def test_forward_event_mute_needs_both_subagent_markers(tmp_path: Path):
    """`agent_id` alone does not mute. Two corroborating markers keep a future
    CC build that stamps one of them on main-session payloads from silencing
    Jarvis entirely — losing the mute is recoverable, losing every event is not."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "false"},
        "session_id": "sess-1",
        "agent_id": "ad419dede82c8f0c1",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row["notification_type"] == "tool_failure"


def test_forward_event_main_session_events_unaffected_by_mute(tmp_path: Path):
    """No agent_id → main-session event → forwarded as usual."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "false"},
        "session_id": "sess-1",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row["notification_type"] == "tool_failure"


def test_forward_event_en_mode_forwards_raw_for_llm_translation(tmp_path: Path):
    """`en` mode with a CHINESE question does NOT pre-bake text; it forwards
    the question payload so the daemon's phrase router calls the LLM to
    translate into Jarvis-toned English. (Same-language questions take the
    verbatim shortcut — see the tests below.)"""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "你想对博客做哪方面的调整",
                    "options": [{"label": "新增博客文章"}, {"label": "调整主题样式"}],
                }
            ],
        },
        "cwd": "/x",
        "session_id": "s1",
    }

    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)

    assert line["notification_type"] == "ask_user_question"
    assert line["tool_name"] == "AskUserQuestion"
    assert line["cwd"] == "/x"
    assert line["session_id"] == "s1"
    assert "text" not in line  # daemon must call router, not synthesize verbatim
    assert "lang" not in line  # daemon picks lang from voice_language config
    # tool_input preserved so the daemon's extract() can summarize it.
    assert line["tool_input"]["questions"][0]["question"] == "你想对博客做哪方面的调整"


def test_forward_event_zh_mode_forwards_raw_for_llm_translation(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [{"question": "Pick a colour", "options": [{"label": "Red"}]}],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="zh") is True
    line = _recv_one(received)
    assert line["notification_type"] == "ask_user_question"
    assert "text" not in line
    assert "lang" not in line


def test_forward_event_en_mode_renders_english_question_verbatim(tmp_path: Path):
    """`en` mode + English question: no LLM round-trip — the hook renders
    question + ALL options verbatim, so nothing is merged or dropped and
    audio isn't 20-60s late (the window in which a cancel kills it)."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "Which fix should we apply?",
                    "options": [
                        {"label": "Skip the LLM"},
                        {"label": "Exempt from cancel"},
                        {"label": "Harden the prompt"},
                    ],
                }
            ],
        },
        "cwd": "/x",
        "session_id": "s1",
    }

    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="en") is True
    line = _recv_one(received)

    assert line["lang"] == "en"
    assert line["text"].startswith("Sir, ")
    assert "Option one: Skip the LLM" in line["text"]
    assert "Option two: Exempt from cancel" in line["text"]
    assert "Option three: Harden the prompt" in line["text"]


def test_forward_event_en_mode_chinese_label_still_goes_to_llm(tmp_path: Path):
    """An English question with a CHINESE option label must NOT verbatim-render
    into the English voice — it needs the LLM translation leg."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "Which theme?",
                    "options": [{"label": "深色模式"}, {"label": "Light"}],
                }
            ],
        },
    }

    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="en") is True
    line = _recv_one(received)

    assert "text" not in line
    assert "lang" not in line
    assert line["tool_input"]["questions"][0]["question"] == "Which theme?"


def test_forward_event_zh_mode_renders_chinese_question_verbatim(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "选哪个方案",
                    "options": [{"label": "方案甲"}, {"label": "方案乙"}],
                }
            ],
        },
    }

    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="zh") is True
    line = _recv_one(received)

    assert line["lang"] == "zh"
    assert line["text"].startswith("先生，")
    assert "选项一: 方案甲" in line["text"]
    assert "选项二: 方案乙" in line["text"]


def test_forward_event_auto_mode_renders_verbatim_for_cjk(tmp_path: Path):
    """`auto` mode keeps the no-LLM shortcut: CJK body → Chinese scaffolding
    + verbatim text, English body → English scaffolding + verbatim text."""
    sock_zh = tmp_path / "zh.sock"
    rcv_zh = _start_unix_echo_server(sock_zh)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [{"question": "选个颜色", "options": [{"label": "红"}]}],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_zh, lang_mode="auto") is True
    line = _recv_one(rcv_zh)
    assert line["lang"] == "zh"
    assert line["text"].startswith("先生，")
    assert "选项一: 红" in line["text"]


def test_forward_event_auto_mode_renders_verbatim_for_english(tmp_path: Path):
    sock_en = tmp_path / "en.sock"
    rcv_en = _start_unix_echo_server(sock_en)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "Pick a colour",
                    "options": [{"label": "Red"}, {"label": "Blue"}],
                }
            ],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_en, lang_mode="auto") is True
    line = _recv_one(rcv_en)
    assert line["lang"] == "en"
    assert line["text"].startswith("Sir, ")
    assert "Option one: Red" in line["text"]


def test_forward_event_auto_mode_does_not_double_prefix_sir(tmp_path: Path):
    """Hook scaffolding only applies in auto mode; the no-double-Sir rule
    only matters there."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "Sir, three threads remain. Which first?",
                    "options": [{"label": "A"}],
                }
            ],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="auto") is True
    line = _recv_one(received)
    assert line["text"].count("Sir,") == 1, line["text"]


def test_forward_event_leaves_non_askuserquestion_payloads_unchanged(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "notification_type": "permission_prompt",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "cwd": "/x",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)
    assert line["notification_type"] == "permission_prompt"
    assert line["tool_name"] == "Bash"
    assert "text" not in line


def test_forward_event_drops_askuserquestion_without_questions(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {"questions": []},
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is False


# ---------------------------------------------------------------------------
# Codex CLI integration
# ---------------------------------------------------------------------------
# Codex's lifecycle-hook payloads share Claude Code's snake_case shape and
# hook_event_name vocabulary for the events both products have in common
# (UserPromptSubmit, PostToolUse). Codex adds PermissionRequest, which has
# no Claude Code analogue and so needs explicit translation. The notify
# command uses a completely different flat payload with kebab-case keys.


def test_forward_event_codex_permission_request_translates_to_permission_prompt(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PermissionRequest",
        "session_id": "sess-xyz",
        "turn_id": "turn-1",
        "cwd": "/Users/me/repo",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_use_id": "tu-1",
        "tool_input": {"command": "rm -rf /tmp/old"},
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "permission_prompt"
    assert row["tool_name"] == "Bash"
    assert row["tool_input"] == {"command": "rm -rf /tmp/old"}
    assert row["cwd"] == "/Users/me/repo"
    assert row["session_id"] == "sess-xyz"


def test_forward_event_codex_first_turn_synthesizes_session_start(tmp_path: Path, monkeypatch):
    """Codex v0.141+ does not fire SessionStart hooks. On the FIRST
    agent-turn-complete for a thread, the hook synthesizes a session_start
    before the idle_prompt so the briefing plays."""
    import jarvis.hook_client as hc

    monkeypatch.setattr(hc, "_CODEX_SESSIONS_DIR", tmp_path / "sessions")
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path, max_conns=2)
    payload = {
        "type": "agent-turn-complete",
        "thread-id": "thr-first",
        "turn-id": "turn-1",
        "cwd": "/Users/me/repo",
        "input-messages": ["please refactor"],
        "last-assistant-message": "Done.",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    rows = _recv_n(received, 2)
    assert rows[0]["notification_type"] == "session_start"
    assert rows[0]["session_id"] is None
    assert rows[0]["cwd"] == "/Users/me/repo"
    assert rows[1]["notification_type"] == "idle_prompt"
    assert rows[1]["session_id"] == "thr-first"


def test_forward_event_codex_second_turn_only_idle_prompt(tmp_path: Path, monkeypatch):
    """After the first turn's synthetic session_start, subsequent turns
    only emit idle_prompt (no repeat briefing)."""
    import jarvis.hook_client as hc

    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(hc, "_CODEX_SESSIONS_DIR", sessions_dir)
    sessions_dir.mkdir()
    (sessions_dir / "thr-seen").write_text("")

    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "type": "agent-turn-complete",
        "thread-id": "thr-seen",
        "turn-id": "turn-9",
        "cwd": "/Users/me/repo",
        "input-messages": ["continue"],
        "last-assistant-message": "Done.",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "idle_prompt"
    assert row["session_id"] == "thr-seen"


def test_forward_event_codex_user_prompt_submit_emits_cancel(tmp_path: Path):
    """Codex's UserPromptSubmit shares CC's name/shape, so cancel behavior
    must work for both without a separate code path."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "sess-codex",
        "cwd": "/x",
        "prompt": "do the thing",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row.get("command") == "cancel"
    assert row.get("session_id") == "sess-codex"


def test_forward_event_codex_post_tool_use_emits_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "sess-codex",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": {"stdout": "..."},
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row.get("command") == "cancel"


def test_forward_event_session_start_startup_becomes_session_start_event(tmp_path: Path):
    """A fresh CC/Codex `SessionStart` (source=startup) triggers the daemon's
    new session_start event so the Iron-Man-style opening briefing speaks."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "sess-9",
        "source": "startup",
        "cwd": "/x",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "session_start"
    assert row["cwd"] == "/x"
    # The briefing is deliberately decoupled from the session's cancel identity:
    # SessionStart is immediately followed by the user's first UserPromptSubmit
    # (a `cancel` for the same session_id), which would otherwise drop the
    # still-composing briefing. Carrying no session_id makes the briefing
    # immune to that cancel so it always speaks. See the regression test below.
    assert row["session_id"] is None
    # Daemon composes the text itself from briefing.py — hook must not pre-bake.
    assert "text" not in row
    assert "lang" not in row


def test_session_start_not_cancelled_by_first_user_prompt(tmp_path: Path):
    """Regression: in Codex (and CC) the user's first prompt fires
    UserPromptSubmit at almost the same instant SessionStart fires, carrying
    the same session_id. UserPromptSubmit translates to a `cancel`; the
    briefing composes via Ollama for 10-40s, so the cancel reliably drops the
    queued/in-flight briefing and the user hears nothing. The fix: the
    briefing event must not share the session's cancel identity, so the
    `cancel` (keyed on the real session_id) can never match it."""
    sid = "019e725d-codex"
    start = {
        "hook_event_name": "SessionStart",
        "source": "startup",
        "session_id": sid,
        "cwd": "/tmp/proj",
    }
    prompt = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": sid,
        "cwd": "/tmp/proj",
        "prompt": "do the thing",
    }

    sock1 = tmp_path / "a.sock"
    rcv1 = _start_unix_echo_server(sock1)
    assert forward_event(io.StringIO(json.dumps(start)), sock1) is True
    briefing = _recv_one(rcv1)

    sock2 = tmp_path / "b.sock"
    rcv2 = _start_unix_echo_server(sock2)
    assert forward_event(io.StringIO(json.dumps(prompt)), sock2) is True
    cancel = _recv_one(rcv2)

    assert briefing["notification_type"] == "session_start"
    assert cancel["command"] == "cancel"
    assert cancel["session_id"] == sid
    # The cancel targets `sid`; the briefing carries no session_id, so the
    # daemon's `cancel_session(sid)` drop/kill (matched by session_id) can
    # never touch the briefing.
    assert briefing["session_id"] != sid
    assert briefing["session_id"] is None


def test_forward_event_session_start_resume_becomes_session_resume(tmp_path: Path):
    """A resumed session gets a short welcome-back (session_resume), not the
    full briefing and not silence. Like the briefing it carries no
    session_id, so the user's first prompt cannot cancel it."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "sess",
        "source": "resume",
        "cwd": "/Users/me/pchat",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "session_resume"
    assert row["session_id"] is None
    assert row["tool_input"] == {"project": "pchat"}


def test_forward_event_session_start_clear_and_compact_are_dropped(tmp_path: Path):
    """SessionStart fires on /clear and compaction too — those stay silent
    rather than greeting whenever the user wipes context."""
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    for source in ("clear", "compact"):
        payload = {
            "hook_event_name": "SessionStart",
            "session_id": "sess",
            "source": source,
            "cwd": "/x",
        }
        assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is False, source


def test_forward_event_session_start_without_source_is_forwarded(tmp_path: Path):
    """Some hook payloads omit `source`; treat that as a cold start so we
    don't silently lose briefings from clients that don't populate the field."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {"hook_event_name": "SessionStart", "session_id": "sess", "cwd": "/x"}
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "session_start"


def test_forward_event_codex_permission_request_with_cancel_disabled(tmp_path: Path):
    """cancel_on_user_action only governs UserPromptSubmit/PostToolUse —
    PermissionRequest is a different axis and must still come through
    even with cancel_on_user_action=False."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PermissionRequest",
        "session_id": "sess",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "cwd": "/x",
    }
    assert forward_event(
        io.StringIO(json.dumps(payload)), sock_path,
        cancel_on_user_action=False,
    ) is True
    row = _recv_one(received)
    assert row["notification_type"] == "permission_prompt"


def test_forward_event_post_tool_use_failure_becomes_tool_failure(tmp_path: Path):
    """CC's PostToolUseFailure → tool_failure event, carrying the tool name,
    the original tool_input, and the error gist under tool_response so the
    daemon's extract_failure() can summarize it."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "session_id": "sess-1",
        "tool_name": "Bash",
        "tool_input": {"command": "npm test"},
        "tool_response": {"error": "3 tests failed"},
        "cwd": "/x",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "tool_failure"
    assert row["tool_name"] == "Bash"
    assert row["tool_input"]["command"] == "npm test"
    assert row["tool_input"]["tool_response"] == {"error": "3 tests failed"}
    assert row["cwd"] == "/x"
    assert "text" not in row  # daemon phrases it, hook does not pre-bake


def test_forward_event_post_tool_use_failure_with_string_response(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_input": {"command": "make"},
        "tool_response": "exit code 2",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "tool_failure"
    assert row["tool_input"]["tool_response"] == "exit code 2"


def test_forward_event_stop_becomes_task_complete(tmp_path: Path):
    """CC's Stop (Claude finished responding) → task_complete event."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "Stop",
        "session_id": "sess-2",
        "cwd": "/x",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "task_complete"
    assert row["tool_name"] is None
    assert row["session_id"] == "sess-2"
    assert row["cwd"] == "/x"


def test_forward_event_subagent_stop_also_becomes_task_complete(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {"hook_event_name": "SubagentStop", "session_id": "s", "cwd": "/x"}
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "task_complete"


def test_forward_event_post_tool_use_still_cancels_not_failure(tmp_path: Path):
    """Regression: a successful PostToolUse must still translate to a cancel,
    not be confused with PostToolUseFailure."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "sess-3",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["command"] == "cancel"


class _ExplodingStream:
    """Stream whose read() raising proves the muted path never consumes stdin."""

    def read(self) -> str:
        raise AssertionError("muted hook must not read its input stream")


def test_forward_event_jarvis_mute_drops_without_reading_stream(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("JARVIS_MUTE", "1")
    ok = forward_event(_ExplodingStream(), tmp_path / "j.sock")
    assert ok is False


def test_forward_event_jarvis_mute_never_touches_socket(monkeypatch, tmp_path: Path):
    """A muted session must be invisible to the daemon — even a session_start
    that would normally always speak gets dropped before the socket connect."""
    monkeypatch.setenv("JARVIS_MUTE", "true")
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {"hook_event_name": "SessionStart", "source": "startup", "cwd": "/x"}
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is False
    import time
    time.sleep(0.05)
    assert received == []


def test_forward_event_jarvis_mute_off_values_still_forward(monkeypatch, tmp_path: Path):
    """"0", "false" (any case), and empty string mean "not muted"."""
    for i, value in enumerate(("0", "false", "FALSE", "")):
        monkeypatch.setenv("JARVIS_MUTE", value)
        sock_path = tmp_path / f"j{i}.sock"
        received = _start_unix_echo_server(sock_path)
        payload = {
            "notification_type": "idle_prompt",
            "tool_name": None,
            "tool_input": {},
            "cwd": "/x",
        }
        assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
        row = _recv_one(received)
        assert row["notification_type"] == "idle_prompt"
