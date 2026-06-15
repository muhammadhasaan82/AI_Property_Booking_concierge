"""
Unit tests for services/config.py and services/pre_router.py changes
introduced in this PR.

Covers:
  - config.py: _dotenv_paths tracking, loading precedence structure,
    _parse_bool, _parse_csv_set helper functions
  - pre_router.py: _normalize, _detect_intent (deferred/disabled paths),
    model resolution via load_model_config (not os.getenv)
  - adk_runner.py: _maybe_handle_search_state_shortcut null safety fixes
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.services.config as config_module
from app.services.pre_router import _detect_intent, _normalize

class TestParseBool:
    def test_true_values(self):
        from app.services.config import _parse_bool
        for val in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON"):
            assert _parse_bool(val) is True, f"Expected True for {val!r}"

    def test_false_values(self):
        from app.services.config import _parse_bool
        for val in ("0", "false", "False", "no", "off", "", "random"):
            assert _parse_bool(val) is False, f"Expected False for {val!r}"

    def test_strips_whitespace(self):
        from app.services.config import _parse_bool
        assert _parse_bool("  true  ") is True
        assert _parse_bool("  false  ") is False


class TestParseCsvSet:
    def test_single_item(self):
        from app.services.config import _parse_csv_set
        assert _parse_csv_set("apartment") == {"apartment"}

    def test_multiple_items(self):
        from app.services.config import _parse_csv_set
        result = _parse_csv_set("apartment, villa, house")
        assert result == {"apartment", "villa", "house"}

    def test_lowercases(self):
        from app.services.config import _parse_csv_set
        result = _parse_csv_set("Apartment,VILLA")
        assert result == {"apartment", "villa"}

    def test_empty_string_returns_empty_set(self):
        from app.services.config import _parse_csv_set
        assert _parse_csv_set("") == set()

    def test_only_commas_returns_empty_set(self):
        from app.services.config import _parse_csv_set
        assert _parse_csv_set(",,,") == set()

    def test_strips_whitespace_around_items(self):
        from app.services.config import _parse_csv_set
        result = _parse_csv_set("  villa  ,  apartment  ")
        assert result == {"villa", "apartment"}



class TestConfigDotenvPaths:
    def test_dotenv_paths_is_a_list(self):
        assert isinstance(config_module._dotenv_paths, list)

    def test_dotenv_paths_contains_only_strings(self):
        for path in config_module._dotenv_paths:
            assert isinstance(path, str), f"Expected string, got {type(path)}: {path!r}"

    def test_repo_root_path_defined(self):
        """_env_root should be at repo_root/.env"""
        env_root = config_module._env_root
        assert isinstance(env_root, Path)
        assert env_root.name == ".env"

    def test_backend_root_path_defined(self):
        """_env_backend should be at backend/.env"""
        env_backend = config_module._env_backend
        assert isinstance(env_backend, Path)
        assert env_backend.name == ".env"

    def test_services_path_defined(self):
        """_env_services should be at services/.env"""
        env_services = config_module._env_services
        assert isinstance(env_services, Path)
        assert env_services.name == ".env"

    def test_backend_root_is_below_repo_root(self):
        """backend/.env should be at a deeper path than repo root .env"""
        assert len(config_module._env_backend.parts) > len(config_module._env_root.parts)

    def test_loading_order_root_before_backend(self):
        """If both root and backend .env paths are loaded, root comes first."""
        paths = config_module._dotenv_paths
        root_str = str(config_module._env_root)
        backend_str = str(config_module._env_backend)
        if root_str in paths and backend_str in paths:
            root_idx = paths.index(root_str)
            backend_idx = paths.index(backend_str)
            assert root_idx < backend_idx, "Root .env must be loaded before backend .env"

    def test_loading_order_backend_before_services(self):
        """If both backend and services .env paths are loaded, backend comes before services."""
        paths = config_module._dotenv_paths
        backend_str = str(config_module._env_backend)
        services_str = str(config_module._env_services)
        if backend_str in paths and services_str in paths:
            backend_idx = paths.index(backend_str)
            services_idx = paths.index(services_str)
            assert backend_idx < services_idx, "backend/.env must be loaded before services/.env"

    def test_three_path_levels_defined_in_module(self):
        """Module defines three potential .env paths."""
        assert hasattr(config_module, "_env_root")
        assert hasattr(config_module, "_env_backend")
        assert hasattr(config_module, "_env_services")

    def test_no_duplicate_paths_in_dotenv_paths(self):
        """Each .env file should appear at most once."""
        paths = config_module._dotenv_paths
        assert len(paths) == len(set(paths)), "Duplicate .env paths detected"



class TestConfigConstants:
    def test_dataset_path_is_str(self):
        assert isinstance(config_module.DATASET_PATH, str)

    def test_mock_mode_is_bool(self):
        assert isinstance(config_module.MOCK_MODE, bool)

    def test_redis_session_ttl_is_int(self):
        assert isinstance(config_module.REDIS_SESSION_TTL_SECONDS, int)

    def test_adk_session_max_events_is_int(self):
        assert isinstance(config_module.ADK_SESSION_MAX_EVENTS, int)

    def test_adk_session_max_context_chars_is_int(self):
        assert isinstance(config_module.ADK_SESSION_MAX_CONTEXT_CHARS, int)



class TestPreRouterNormalize:
    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_strips_whitespace(self):
        assert _normalize("  hello  ") == "hello"

    def test_collapses_internal_spaces(self):
        assert _normalize("hello   world") == "hello world"

    def test_removes_punctuation(self):
        assert _normalize("hello, world!") == "hello world"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_only_punctuation(self):
        assert _normalize("!!! ???") == ""

    def test_non_string_is_cast(self):
        assert _normalize(42) == "42"

    def test_mixed_case_punctuation_spaces(self):
        assert _normalize("  What's   Up?  ") == "whats up"



class TestPreRouterDetectIntent:
    def test_returns_none_when_pre_router_disabled(self):
        """When cfg.pre_router is not enabled, _detect_intent returns None."""
        mock_cfg = MagicMock()
        mock_cfg.pre_router = MagicMock()
        mock_cfg.pre_router.enabled = False

        with patch("app.services.pre_router.cfg", mock_cfg):
            result = _detect_intent("hello")
        assert result is None

    def test_returns_none_when_pre_router_missing(self):
        """When cfg.pre_router is None, _detect_intent returns None."""
        mock_cfg = MagicMock()
        mock_cfg.pre_router = None

        with patch("app.services.pre_router.cfg", mock_cfg):
            result = _detect_intent("hello")
        assert result is None

    def test_returns_none_when_no_intents_configured(self):
        """When intents is None/empty, _detect_intent returns None."""
        mock_cfg = MagicMock()
        mock_cfg.pre_router.enabled = True
        mock_cfg.pre_router.intents = None

        with patch("app.services.pre_router.cfg", mock_cfg):
            result = _detect_intent("hello")
        assert result is None

    def test_returns_none_when_no_match(self):
        """When no intent matches the message, returns None (explicit return added in PR)."""
        mock_cfg = MagicMock()
        mock_cfg.pre_router.enabled = True

        intent_cfg = MagicMock()
        intent_cfg.match = MagicMock()
        intent_cfg.match.normalized_exact = []
        intent_cfg.match.normalized_starts_with = []
        intent_cfg.match.normalized_contains_any = []
        intent_cfg.defer_to_adk = False

        intents_obj = MagicMock()
        intents_obj.__iter__ = MagicMock(return_value=iter([]))
        mock_cfg.pre_router.intents = intents_obj

        with patch("app.services.pre_router.cfg", mock_cfg):
            result = _detect_intent("some random message")
        assert result is None

    def test_defer_to_adk_returns_none(self):
        """When an intent matches but has defer_to_adk=True, must return None."""
        mock_cfg = MagicMock()
        mock_cfg.pre_router.enabled = True

        class _IntentCfg:
            pass

        class _MatchCfg:
            normalized_exact = ["hello"]
            normalized_starts_with = []
            normalized_contains_any = []

        ic = _IntentCfg()
        ic.match = _MatchCfg()
        ic.defer_to_adk = True

        class _Intents:
            hello_intent = ic

        mock_cfg.pre_router.intents = _Intents()

        with patch("app.services.pre_router.cfg", mock_cfg):
            result = _detect_intent("hello")
        assert result is None



class TestPreRouterModelResolution:
    def test_generate_reply_uses_model_config(self, monkeypatch):
        """_generate_reply must call load_model_config().pre_router_fast_model (not os.getenv)."""
        from app.config.model_config_loader import ResolvedModelConfig

        expected_model = "groq/llama-custom"
        mock_config = ResolvedModelConfig(
            dispatcher_model="openai/gpt-5-nano",
            voice_model="groq/llama-3.1-8b-instant",
            pre_router_fast_model=expected_model,
            mem0_llm_model="groq/llama-3.1-8b-instant",
            mem0_llm_provider="groq",
        )

        models_called = []

        def _mock_load(raw=None):
            models_called.append(True)
            return mock_config

        with patch("app.services.pre_router.load_model_config", _mock_load):
            with patch("litellm.completion") as mock_completion:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = "test reply"
                mock_completion.return_value = mock_resp

                import asyncio

                mock_cfg = MagicMock()
                mock_cfg.pre_router.generator.temperature = 0.7
                mock_cfg.pre_router.generator.max_tokens = 80
                mock_cfg.pre_router.generator.timeout_seconds = 4
                mock_cfg.pre_router.emergency_fallback = ""
                mock_cfg.pre_router.intents = MagicMock()

                with patch("app.services.pre_router.cfg", mock_cfg):
                    asyncio.run(
                        __import__(
                            "app.services.pre_router", fromlist=["_generate_reply"]
                        )._generate_reply("test_intent", "hello")
                    )

        assert len(models_called) > 0, "_generate_reply did not call load_model_config()"




class TestMaybeHandleSearchStateShortcut:
    """Tests for the null-safety fixes in _maybe_handle_search_state_shortcut."""

    @pytest.mark.asyncio
    async def test_returns_none_when_snapshot_is_none(self):
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _maybe_handle_search_state_shortcut(
                session_id="test-session", message="show me more"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_snapshot_is_not_dict(self):
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value="not-a-dict",
        ):
            result = await _maybe_handle_search_state_shortcut(
                session_id="s", message="show me more"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_state_is_not_dict(self):
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value={"state": "not-a-dict"},
        ):
            result = await _maybe_handle_search_state_shortcut(
                session_id="s", message="show me more"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_soft_state_is_not_dict(self):
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value={"state": {"soft_state": "not-a-dict"}},
        ):
            result = await _maybe_handle_search_state_shortcut(
                session_id="s", message="show me more"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_shortcut_match(self):
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        snapshot = {
            "state": {"soft_state": {}},
        }

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ):
            result = await _maybe_handle_search_state_shortcut(
                session_id="s", message="completely unrelated message"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_payload_is_empty(self):
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        snapshot = {
            "state": {
                "soft_state": {"all_search_results": [{"id": "1"}]}
            }
        }

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ):
            with patch("app.agents.tools.search.paginate_stored_results", return_value=None):
                result = await _maybe_handle_search_state_shortcut(
                    session_id="s", message="show me more"
                )
        assert result is None

    @pytest.mark.asyncio
    async def test_paginate_action_returns_payload(self):
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        expected_payload = {"status": "properties_found", "total_found": 10}
        snapshot = {
            "state": {
                "soft_state": {
                    "all_search_results": [{"id": str(i)} for i in range(10)],
                    "current_page": 1,
                    "page_size": 5,
                }
            },
            "history": [],
            "meta": {"app_name": "test", "user_id": "u1"},
        }

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ):
            with patch("app.agents.tools.search.paginate_stored_results", return_value=expected_payload):
                with patch(
                    "app.services.adk_runner.save_session_snapshot",
                    new_callable=AsyncMock,
                ) as mock_save:
                    result = await _maybe_handle_search_state_shortcut(
                        session_id="s", message="show me more"
                    )

        assert result == expected_payload
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_snapshot_called_with_meta_keys_only(self):
        """save_session_snapshot is called with only known meta keys."""
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        payload = {"status": "properties_found"}
        snapshot = {
            "state": {
                "soft_state": {
                    "all_search_results": [{"id": "1"}],
                }
            },
            "history": [{"role": "user", "content": "hi"}],
            "meta": {
                "app_name": "concierge",
                "user_id": "user-123",
                "last_update_time": "2024-01-01T00:00:00Z",
                "internal_secret": "should-not-be-passed",
            },
        }

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ):
            with patch("app.agents.tools.search.paginate_stored_results", return_value=payload):
                with patch(
                    "app.services.adk_runner.save_session_snapshot",
                    new_callable=AsyncMock,
                ) as mock_save:
                    await _maybe_handle_search_state_shortcut(
                        session_id="s", message="show me more"
                    )

        call_kwargs = mock_save.call_args[1] if mock_save.call_args else {}
        metadata = call_kwargs.get("metadata", {})
        assert "internal_secret" not in metadata
        assert "app_name" in metadata
        assert "user_id" in metadata

    @pytest.mark.asyncio
    async def test_state_soft_state_is_none_returns_none(self):
        """soft_state=None (not missing, but explicitly None) triggers None guard."""
        from app.services.adk_runner import _maybe_handle_search_state_shortcut

        snapshot = {
            "state": {"soft_state": None},
        }

        with patch(
            "app.services.adk_runner.get_session_snapshot",
            new_callable=AsyncMock,
            return_value=snapshot,
        ):
            result = await _maybe_handle_search_state_shortcut(
                session_id="s", message="show me more"
            )
        assert result is None



class TestConfigDotenvPathValidity:
    def test_env_root_path_is_absolute(self):
        assert config_module._env_root.is_absolute()

    def test_env_backend_path_is_absolute(self):
        assert config_module._env_backend.is_absolute()

    def test_env_services_path_is_absolute(self):
        assert config_module._env_services.is_absolute()

    def test_backend_root_is_parents_2_of_config(self):
        """_backend_root is 2 levels up from services/config.py."""
        services_file = Path(config_module.__file__).resolve()
        expected_backend = services_file.parents[1]
        assert config_module._backend_root == expected_backend

    def test_repo_root_is_parents_3_of_config(self):
        """_repo_root is 3 levels up from services/config.py."""
        services_file = Path(config_module.__file__).resolve()
        expected_repo = services_file.parents[2]
        assert config_module._repo_root == expected_repo