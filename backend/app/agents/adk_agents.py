# app/agents/adk_agents.py
"""
ADK 2.0 — Native V2 Agentic Architecture

Dual-Model Architecture:
  Node 1 (triage_router)   → Dispatcher model (temperature driven by config)
  Node 2 (concierge_voice) → Voice model       (temperature driven by config)

The SequentialAgent pipeline: triage_router → concierge_voice.

FILE SIZE POLICY — this file contains ONLY agent wiring.
┌─────────────────────────────────────────────────────────────────┐
│  To change ANY behaviour — edit app/config/agent_config.yaml   │
│  To change prompts       — edit app/prompts/*.md               │
│  NO hardcoded values exist in this file.                        │
└─────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import os

import litellm
from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types as genai_types

from app.config.agent_config_loader import cfg
from app.agents.prompts.loader import load_prompt

# Disable LiteLLM telemetry
litellm.telemetry = False
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOG"] = "ERROR"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expose model names for sub-modules that need them (no circular import)
# Both come from cfg which reads env vars with YAML as fallback.
# ---------------------------------------------------------------------------
DISPATCHER_MODEL: str = cfg.dispatcher_model
VOICE_MODEL: str = cfg.voice_model

# ---------------------------------------------------------------------------
# Dual-Model Backends (via LiteLLM — no Google Cloud dependency)
# ---------------------------------------------------------------------------
dispatcher_llm = LiteLlm(model=DISPATCHER_MODEL)
voice_llm = LiteLlm(model=VOICE_MODEL)

# ---------------------------------------------------------------------------
# Generation configs — temperatures from agent_config.yaml
# ---------------------------------------------------------------------------
DISPATCHER_CONFIG = genai_types.GenerateContentConfig(
    temperature=cfg.dispatcher_temperature,
)
VOICE_CONFIG = genai_types.GenerateContentConfig(
    temperature=cfg.voice_temperature,
)

# ---------------------------------------------------------------------------
# Prompt loading — prompts live in app/prompts/*.md
# Editable without touching Python.
# ---------------------------------------------------------------------------
TRIAGE_INSTRUCTION: str = load_prompt("triage_instruction.md")
VOICE_INSTRUCTION: str = load_prompt("voice_instruction.md")

# ---------------------------------------------------------------------------
# Tool imports — all tool functions live in their respective sub-modules
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
