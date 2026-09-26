"""Fallback phrase templates when all LLM providers fail.

Tone tries to approximate Jarvis: polite, brief, with "Sir"/"先生".
"""
from __future__ import annotations

from ..types import Event, Lang

_ZH: dict[str, str] = {
    "permission_prompt": "先生，Claude 请求使用 {tool} 的权限。",
    "idle_prompt": "先生，Claude 正在等候您的指示。",
    "elicitation_dialog": "先生，有个对话框等您填写。",
    "ask_user_question": "先生，有个选择题等您拍板。",
    # session_start normally bypasses the LLM router and is composed by
    # briefing.py; this template is only reached if that path errors out.
    "session_start": "先生，欢迎回来。",
    "session_resume": "先生，欢迎回来。",
    "tool_failure": "先生，{tool} 执行失败了。",
    "task_complete": "先生，已完成。",
    # Tier 1
    "context_compacting": "先生，对话上下文即将被压缩。",
    "rate_limited": "先生，达到了速率限制，稍作等候。",
    "subagent_spawned": "先生，已派出一个子代理。",
    "max_turns_reached": "先生，已达到轮次上限——Claude 已停止。",
    # Tier 2
    "api_error": "先生，API 报错了。",
    "session_end": "先生，下次再见。",
    "context_compacted": "上下文已压缩，先生。继续。",
    "context_overflow": "先生，上下文窗口已满。",
}

_EN: dict[str, str] = {
    "permission_prompt": "Sir, Claude requests permission for {tool}.",
    "idle_prompt": "Sir, Claude awaits your guidance.",
    "elicitation_dialog": "Sir, a dialog awaits your input.",
    "ask_user_question": "Sir, a question awaits your decision.",
    "session_start": "At your service, sir.",
    "session_resume": "Welcome back, sir.",
    "tool_failure": "Sir, {tool} has failed.",
    "task_complete": "All done, sir.",
    # Tier 1
    "context_compacting": "Sir, the conversation context is about to be compressed.",
    "rate_limited": "Sir, we have hit the rate limit — a brief intermission.",
    "subagent_spawned": "Sir, a sub-agent has been dispatched.",
    "max_turns_reached": "Sir, the turn limit has been reached — Claude has stopped.",
    # Tier 2
    "api_error": "Sir, the API has returned an error.",
    "session_end": "Until next time, sir.",
    "context_compacted": "Context compacted, sir. We carry on.",
    "context_overflow": "Sir, the context window is full.",
}


def render_template(event: Event, lang: Lang) -> str:
    table = _ZH if lang == "zh" else _EN
    tmpl = table.get(event.notification_type, table["idle_prompt"])
    return tmpl.format(tool=event.tool_name or "something")
