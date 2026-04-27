"""
app/config/agent_config_loader.py
-----------------------------------
Loads app/config/agent_config.yaml once at startup and makes it available as a
single frozen config object throughout the application.

Environment variables always WIN over YAML values where both are defined.
No value in this system needs to be changed in Python source code.

Usage:
    from app.config.agent_config_loader import cfg

    # All values accessed as dot-notation attributes
    cfg.session.cache_ttl_seconds
    cfg.booking.required_fields
    cfg.messages.resolution_unresolved_default
    cfg.status.properties_found
    cfg.intent_routing.history_action_intents   # returns frozenset
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, FrozenSet

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "agent_config.yaml"

class _Namespace:
    """Recursive dot-access wrapper over a plain dict."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Namespace(value))
            elif isinstance(value, list):
                setattr(self, key, value)
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, "").strip() or default

def _load() -> "_AgentConfig":
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    return _AgentConfig(raw)


class _AgentConfig:
    """
    Typed config facade built on top of agent_config.yaml.
    All scalar values support environment variable overrides.
    List values (frozensets, lists) are read from YAML only.
    """

    def __init__(self, raw: dict) -> None:
        self._raw = raw

        m = raw["models"]
        self.dispatcher_model: str = _env_str(
            m["dispatcher"]["env_key"], m["dispatcher"]["default"]
        )
        self.dispatcher_temperature: float = m["dispatcher"]["temperature"]
        self.voice_model: str = _env_str(
            m["voice"]["env_key"], m["voice"]["default"]
        )
        self.voice_temperature: float = m["voice"]["temperature"]

        s = raw["session"]
        self.session_ttl: int = _env_int(
            "SOFT_SESSION_CACHE_TTL_SECONDS", s["cache_ttl_seconds"]
        )
        self.engagement_fatigued_turns: int = _env_int(
            "ENGAGEMENT_FATIGUED_TURNS", s["unresolved_turns_fatigued"]
        )
        self.engagement_exhausted_turns: int = _env_int(
            "ENGAGEMENT_EXHAUSTED_TURNS", s["unresolved_turns_exhausted"]
        )

        sr = raw["search"]
        self.rerank_limit: int = _env_int(
            "PROPERTY_RERANK_LIMIT", sr["rerank_limit"]
        )
        self.rerank_timeout: float = _env_float(
            "PROPERTY_RERANK_TIMEOUT_SECONDS", sr["rerank_timeout_seconds"]
        )
        self.search_result_limit: int = _env_int(
            "PROPERTY_SEARCH_RESULT_LIMIT", sr.get("result_limit", 5)
        )
        self.search_result_limit_max: int = _env_int(
            "PROPERTY_SEARCH_RESULT_LIMIT_MAX", sr.get("result_limit_max", 10)
        )
        self.search_summary_mode_threshold: int = _env_int(
            "PROPERTY_SEARCH_SUMMARY_THRESHOLD", sr.get("summary_mode_threshold", 12)
        )

        ds = raw["dataset"]
        self.dataset_relative_path: str = ds["relative_path"]
        self.city_column_candidates: list[str] = ds["city_column_candidates"]

        ir = raw["intent_routing"]
        self.history_action_intents: FrozenSet[str] = frozenset(
            ir["history_action_intents"]
        )
        self.new_search_action_intents: FrozenSet[str] = frozenset(
            ir["new_search_action_intents"]
        )

        bk = raw["booking"]
        self.date_format: str = bk["date_format"]
        self.booking_required_fields: list[str] = bk["required_fields"]
        self.booking_required_numeric_fields: list[str] = bk["required_numeric_fields"]
        self.booking_source_tag: str = bk["booking_source_tag"]
        self.booking_confirmed_status: str = bk["booking_confirmed_status"]

        st = raw["small_talk"]
        self.small_talk_valid_types: FrozenSet[str] = frozenset(st["valid_types"])
        self.small_talk_default_type: str = st["default_type"]

        en = raw["engagement"]
        self.engagement_valid_states: FrozenSet[str] = frozenset(
            en["valid_states"]
        )

        res = raw["resolution"]
        self.resolution_valid_intents: FrozenSet[str] = frozenset(
            res["valid_intent_classes"]
        )
        self.resolution_fallback_intent: str = res["fallback_intent"]

        msg = raw["messages"]
        self.msg_resolution_default: str = msg["resolution_unresolved_default"]
        self.msg_resolution_frustrated: str = msg["resolution_unresolved_frustrated"]
        self.msg_resolution_not_matched_log: str = msg["resolution_not_matched_log"]
        self.msg_selection_out_of_range: str = msg.get(
            "selection_out_of_range",
            "That option number is outside the current shortlist."
        )
        self.msg_escalation_default: str = msg["escalation_default_reason"]

        st_raw = raw["status"]
        self.status = _Namespace(st_raw)

        src_raw = raw["sources"]
        self.source = _Namespace(src_raw)

        an = raw.get("anomaly", {})
        self.anomaly_tool_loop_threshold: int = _env_int(
            "ANOMALY_TOOL_LOOP_THRESHOLD", an.get("tool_loop_threshold", 5)
        )
        self.anomaly_time_window_seconds: int = _env_int(
            "ANOMALY_TIME_WINDOW_SECONDS", an.get("time_window_seconds", 30)
        )
        self.anomaly_session_ttl_minutes: int = _env_int(
            "ANOMALY_SESSION_TTL_MINUTES", an.get("session_ttl_minutes", 30)
        )
        self.anomaly_fallback_message: str = an.get(
            "fallback_message",
            "I seem to be having a bit of trouble processing that request. "
            "Could you try rephrasing or providing a few more details?"
        )
        self.anomaly_exempt_tools: FrozenSet[str] = frozenset(
            an.get("exempt_tools", [])
        )
        ft = raw.get("features", {})
        self.feature_understanding_frame: bool = _env_str(
            "UNDERSTANDING_FRAME_ENABLED",
            "1" if ft.get("understanding_frame_enabled", True) else "0",
        ).lower() in {"1", "true", "yes"}
        self.feature_policy_router_mode: str = _env_str(
            "POLICY_ROUTER_MODE",str(ft.get("policy_router_mode", "off")).lower()
        )
        self.feature_response_policies: bool = _env_str(
            "RESPONSE_POLICIES_ENABLED",
            "1" if ft.get("response_policies_enabled", True) else "0",
        ).lower() in {"1", "true", "yes"}
        self.feature_tool_registry: bool=_env_str(
            "TOOL_REGISTRY_ENABLED",
            "1" if ft.get("tool_registry_enabled", True)else "0"
        ).lower() in {"1", "true","yes"}

    def classify_engagement(self, unresolved_turns: int) -> str:
        """
        Deterministic engagement classifier driven by config thresholds.
        Thresholds are loaded from agent_config.yaml (or ENV overrides).
        No hardcoded numbers in Python.
        """
        if unresolved_turns >= self.engagement_exhausted_turns:
            return "exhausted_or_frustrated"
        if unresolved_turns >= self.engagement_fatigued_turns:
            return "fatigued"
        return "engaged"

    def get_dataset_path(self, backend_root: Path) -> Path:
        """Resolve the dataset path relative to the backend root."""
        return backend_root / self.dataset_relative_path

cfg: _AgentConfig = _load()
