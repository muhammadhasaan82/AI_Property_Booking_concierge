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

from doctest import OutputChecker
import logging
import os

from app.agents.tools.search import get_all_available_cities, search_properties
import litellm
from google.adk.agents import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types as genai_types

from app.config.agent_config_loader import cfg
from app.agents.prompts.loader import load_prompt
from app.config.tool_registry_loader import registry as _tool_registry

litellm.telemetry = False
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOG"] = "ERROR"

logger = logging.getLogger(__name__)

DISPATCHER_MODEL: str = cfg.dispatcher_model
VOICE_MODEL: str = cfg.voice_model

dispatcher_llm = LiteLlm(model=DISPATCHER_MODEL)
voice_llm = LiteLlm(model=VOICE_MODEL)

DISPATCHER_CONFIG = genai_types.GenerateContentConfig(
    temperature=cfg.dispatcher_temperature,
)
VOICE_CONFIG = genai_types.GenerateContentConfig(
    temperature=cfg.voice_temperature,
)

TRIAGE_INSTRUCTION: str = load_prompt("triage_instruction.md")
VOICE_INSTRUCTION: str = load_prompt("voice_instruction.md")

if cfg.feature_tool_registry and _tool_registry.tools:
    _resolved_tools = list(_tool_registry.resolve_callables().values())
else:
    from .tools.support import handle_small_talk, check_faq, check_booking_status, escalate_to_human
    from .tools.search import (
        get_all_available_cities,
        search_properties,
        get_property_details,
        select_property,
    )
    from .tools.booking import(
        request_booking_details,
        review_booking_details,
        process_v2_booking,
    )
    _resolved_tools = [
        handle_small_talk,
        search_properties,
        select_property,
        get_property_details,
        check_faq,
        check_booking_status,
        request_booking_details,
        review_booking_details,
        process_v2_booking,
        escalate_to_human,
        get_all_available_cities,
    ]

logger.info(
    "[adk_agents] triage_router built with %d tools (registry=%s)",
    len(_resolved_tools),
    cfg.feature_tool_registry and bool(_tool_registry.tools),
)

triage_router = LlmAgent(
    model=dispatcher_llm,
    name="triage_router",
    description="Routes user intent to the correct tool. Does not generate conversational text.",
    instruction=TRIAGE_INSTRUCTION,
    tools=_resolved_tools,
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


root_agent = SequentialAgent(
    name="concierge_pipeline",
    sub_agents=[triage_router, concierge_voice],
    description="AI Property Booking Concierge — routes user intent and generates responses.",
)
