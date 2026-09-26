from pathlib import Path

import pytest

from jarvis.config import load_config


def test_humor_level_clamps_out_of_range_values(tmp_path: Path):
    """A typo in user TOML must not leave the daemon refusing to start."""
    p = tmp_path / "c.toml"
    p.write_text("[behavior]\nhumor_level = 99\n")
    assert load_config(p).behavior.humor_level == 3
    p.write_text("[behavior]\nhumor_level = -7\n")
    assert load_config(p).behavior.humor_level == 0
    p.write_text("[behavior]\nhumor_level = 2\n")
    assert load_config(p).behavior.humor_level == 2


def test_load_config_returns_defaults_when_file_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.llm.provider == "deepseek"
    assert cfg.llm.fallback == "ollama"
    assert cfg.tts.provider == "xtts"
    # Bounds PyTorch's MPS allocator, which otherwise grows to fill free RAM
    # and never gives it back — 13 GB after a day, 23.5 GB for one long line.
    # The batch decode path holds 0.2; the old streaming path OOMed there and
    # drops to the `say` voice. See XTTSConfig for the full measurement.
    assert cfg.tts.xtts.mps_memory_ratio == 0.2
    assert cfg.behavior.dedup_window_seconds == 10
    assert cfg.behavior.queue_max_size == 5
    assert cfg.behavior.voice_language == "en"
    assert cfg.behavior.events == [
        "permission_prompt",
        "idle_prompt",
        "elicitation_dialog",
        "ask_user_question",
        "session_start",
        "session_resume",
        "tool_failure",
        "context_compacting",
        "rate_limited",
        "subagent_spawned",
        "max_turns_reached",
    ]
    # task_complete (CC Stop) fires after every assistant turn, so it stays
    # opt-in.  Tier 2 events (api_error, session_end, context_compacted,
    # context_overflow) are also opt-in.
    assert "task_complete" not in cfg.behavior.events
    for t2 in ("api_error", "session_end", "context_compacted", "context_overflow"):
        assert t2 not in cfg.behavior.events
    # session_briefing block defaults in lockstep with the install.py TOML.
    assert cfg.behavior.session_briefing.enabled is True
    assert cfg.behavior.session_briefing.city == ""
    assert cfg.behavior.session_briefing.weather_ttl_seconds == 600
    assert cfg.behavior.session_briefing.min_interval_seconds == 0
    # humor_level defaults to 1 (light wit). Clamped on load.
    assert cfg.behavior.humor_level == 1
    # Stale-drop floor: LLM-phrased events older than this at dequeue time
    # are dropped instead of spoken. 0 disables.
    assert cfg.behavior.stale_event_max_age_seconds == 60.0


def test_load_config_reads_toml(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(
        """
[llm]
provider = "anthropic"
fallback = "deepseek"

[behavior]
dedup_window_seconds = 30
queue_max_size = 9
"""
    )
    cfg = load_config(p)
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.fallback == "deepseek"
    assert cfg.behavior.dedup_window_seconds == 30
    assert cfg.behavior.queue_max_size == 9
    # Untouched fields stay default
    assert cfg.tts.provider == "xtts"


def test_load_config_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.paths.socket.startswith(str(tmp_path))


def test_load_config_defaults_phrase_budget(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.behavior.phrase_target_chars == 70
    assert cfg.behavior.phrase_hard_cap == 120
    assert cfg.behavior.privacy.cloud_redaction is True
    # legacy field kept silently for back-compat
    assert hasattr(cfg.behavior, "phrase_max_chars")


def test_behavior_default_has_cancel_on_user_action_true(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("")
    cfg = load_config(cfg_path)
    assert cfg.behavior.cancel_on_user_action is True


def test_behavior_cancel_on_user_action_overridable(tmp_path: Path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text("[behavior]\ncancel_on_user_action = false\n")
    cfg = load_config(cfg_path)
    assert cfg.behavior.cancel_on_user_action is False


def test_load_config_reads_privacy_override(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(
        """
[behavior]
phrase_target_chars = 90
phrase_hard_cap = 160

[behavior.privacy]
cloud_redaction = false
"""
    )
    cfg = load_config(p)
    assert cfg.behavior.phrase_target_chars == 90
    assert cfg.behavior.phrase_hard_cap == 160
    assert cfg.behavior.privacy.cloud_redaction is False
