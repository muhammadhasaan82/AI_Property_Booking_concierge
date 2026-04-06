# app/agents/adk_agents.py
"""
ADK 2.0 — Native V2 Agentic Architecture

Dual-Model Architecture:
  Node 1 (triage_router)   → GPT-5 Nano via LiteLLM  (temperature=1)
  Node 2 (concierge_voice) → Llama-3.3-70B via Groq   (temperature=0.6)

The SequentialAgent pipeline: triage_router → concierge_voice.

File size policy:
  This file contains ONLY agent wiring and configuration.
  All tool functions live in app/agents/tools/
  All prompts live in app/prompts/*.md
  All status constants live in app/agents/status_codes.py
  All shared helpers live in app/agents/tools/helpers.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import litellm
from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types as genai_types

# Disable LiteLLM telemetry at Python level
litellm.telemetry = False
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOG"] = "ERROR"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or str(default)).strip()
    try:
        return float(raw)
    except Exception:
        return default


DISPATCHER_MODEL: str = os.getenv("ADK_DISPATCHER_MODEL", "openai/gpt-5-nano")
VOICE_MODEL: str = os.getenv("ADK_VOICE_MODEL", "groq/llama-3.3-70b-versatile")

# Propagate env-overridable config to helpers module
from .tools.helpers import SOFT_SESSION_TTL_SECONDS as _DEFAULT_TTL  # noqa: E402
import app.agents.tools.helpers as _helpers_mod  # noqa: E402
_helpers_mod.SOFT_SESSION_TTL_SECONDS = _env_int("SOFT_SESSION_CACHE_TTL_SECONDS", _DEFAULT_TTL)

# ---------------------------------------------------------------------------
# Dual-Model Backends (via LiteLLM — no Google Cloud dependency)
# ---------------------------------------------------------------------------
dispatcher_llm = LiteLlm(model=DISPATCHER_MODEL)
voice_llm = LiteLlm(model=VOICE_MODEL)

# ---------------------------------------------------------------------------
# Generation configs
#
# Two-Speed Streaming Rule:
#   DISPATCHER_CONFIG — triage_router token stream is SILENTLY CONSUMED by runner.
#   VOICE_CONFIG      — concierge_voice text deltas are STREAMED to the UI.
# ---------------------------------------------------------------------------
DISPATCHER_CONFIG = genai_types.GenerateContentConfig(temperature=1)
VOICE_CONFIG = genai_types.GenerateContentConfig(temperature=0.6)

# ---------------------------------------------------------------------------
# Prompt loading
# Prompts live in app/prompts/*.md so they can be edited without touching Python.
# ---------------------------------------------------------------------------
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

TRIAGE_INSTRUCTION: str = (_PROMPTS_DIR / "triage_instruction.md").read_text(encoding="utf-8")
VOICE_INSTRUCTION: str = (_PROMPTS_DIR / "voice_instruction.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Tool imports
# All tool functions are defined in their respective sub-modules.
# ---------------------------------------------------------------------------
from .tools.search import (  # noqa: E402
    get_all_available_cities,
    search_properties,
    get_property_details,
)
from .tools.booking import (  # noqa: E402
    request_booking_details,
    review_booking_details,
    process_v2_booking,
)
from .tools.support import (  # noqa: E402
    handle_small_talk,
    check_faq,
    check_booking_status,
    escalate_to_human,
)

# ---------------------------------------------------------------------------
# ADK Agent Nodes
# ---------------------------------------------------------------------------

triage_router = LlmAgent(
    model=dispatcher_llm,
    name="triage_router",
    description="Routes user intent to the correct tool. Does not generate conversational text.",
    instruction=TRIAGE_INSTRUCTION,
    tools=[
        handle_small_talk,
        search_properties,
        get_property_details,
        check_faq,
        check_booking_status,
        request_booking_details,
        review_booking_details,
        process_v2_booking,
        escalate_to_human,
        get_all_available_cities,
    ],
    output_key="router_output",
    generate_content_config=DISPATCHER_CONFIG,
)

concierge_voice = LlmAgent(
    model=voice_llm,
    name="concierge_voice",
    description="Formats tool outputs into warm, human-like responses.",
    instruction=VOICE_INSTRUCTION,
    output_key="final_reply",
    generate_content_config=VOICE_CONFIG,
)

# ---------------------------------------------------------------------------
# Sequential Pipeline (The V2 Brain)
# ---------------------------------------------------------------------------

root_agent = SequentialAgent(
    name="concierge_pipeline",
    sub_agents=[triage_router, concierge_voice],
    description="AI Property Booking Concierge — routes user intent and generates responses.",
)
