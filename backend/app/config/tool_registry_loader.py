"""
Loads tool_registry.yaml and provides:
  - registry.tools           dict[str, ToolSpec]
  - registry.get(name)       lookup by tool name
  - registry.for_intent(...)  list of tools allowed for a given intent
  - registry.resolve_callables()  import each module/function for ADK wiring
 
Phase 1: agents/adk_agents.py uses resolve_callables() to build its tool list,
so adding a tool no longer requires Python edits in adk_agents.py.
"""
from __future__ import annota   
import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_REGISTRY_PATH = Path(__file__).resolve().parent/"tool_registry.yaml"

class ToolSpec(BaseModel):
    name: str
    module: str
    function: str
    description: str
    intent: List[str] = Field(default_factory=list)
    required_inputs: List[str] = Field(default_factory=list)
    optional_inputs: List[str] = Field(default_factory=list)
    requires_context: List[str] = Field(default_factory=list)
    source_priority: List[str] = Field(default_factory=list)
    response_priority: List[str] = Field(default_factory=list)
    response_policy: Optional[str] = None
    rerank: bool = False
    schema_ref: Optional[str] = None
    missing_input_strategy: Optional[str] = None
    requires_explicit_user_authorization = False

class _ToolRegistry:
    def __init__(self, raw: Dict[str, Any]) -> None:
        self.version: str = str(raw.get("version", "1.0"))
        self.tools: Dict[str, ToolSpec] = {}
        for name, body in (raw.get("tools") or {}).items():
            try:
                self.tools[name] = ToolSpec(name=name, **body)
            except Exception as exc:
                logger.error("[tool_registry] invalid spec for %r: %s", name, exc)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.tools.get(name)

    def for_intent(self, intent: str) -> List[ToolSpec]:
        return [t for t in self.tools.values() if intent in t.intents]

    def resolve_callables(self) -> Dict[str, Callable]:
        """Import each tool's function. Returns {tool_name: callable}."""
        callables: Dict[str, Callable[..., Any]] = {}
        for name, spec in self.tools.items():
            try:
                module = importlib.import_module(spec.module)
                fn = getattr(module, spec.function)
                callables[name] = fn
            except Exception as exc:
                logger.error("[tool_registry] failed to import %s.%s: %s", spec.module, spec.function, exc)
        return callables

def _load() -> _ToolRegistry:
    if not _REGISTRY_PATH.exists():
        logger.warning("[tool_registry] %s missing, registry empty", _REGISTRY_PATH)
        return _ToolRegistry({})
    with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _ToolRegistry(raw)

registry: _ToolRegistry = _load()

def reload() -> None:
    global registry
    registry = _load()
    logger.info("[tool_registry] reloaded, %d tools available", len(registry.tools))
