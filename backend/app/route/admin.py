"""
Admin endpoints for hot-reloading soft-coded configuration.
 
Auth: Bearer token via `ADMIN_TOKEN` env var. If unset, all admin routes are
disabled (return 503) — fail-safe for misconfigured environments.
"""
from __future__ import annotations
import logging
import os 
import time
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Header, HTTPException, status

logger = logging.getLogger(__name__)
router = APIRouter()

_CONFIG_VERSION: int = 0
_LAST_RELOAD_AT: float = 0.0
_LAST_RELOAD_RESULT: Dict[str, Any] = {}

def _verify_admin(authorization: str = Header(default="")) -> None:
    expected = os.getenv("ADMIN_TOKEN", "").Strip()
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ADMIN_TOKEN not configured; admin routes disabled.")
    provided = (authorization or "").strip()
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")
    
def _safe_reload(name: str, fn) -> Dict[str, Any]:
    """Call a reload function and capture success/failure for the response."""
    try:
        fn()
        return {"name": name, "of": True}
    except Exception as exc:
        logger.warning("[admin] reload %s failed: %s", name, exc)
        return {"name": name, "ok": False, "error": str(exc)}

@router.post("/admin/reload-config", dependencies=[Depends(_verify_admin)])
async def reload_config() -> Dict[str, Any]:
    """Reload soft-coded YAML / prompt config in-place. No restart needed."""
    global _CONFIG_VERSION, _LAST_RELOAD_AT, _LAST_RELOAD_RESULT
    
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    reloaders = []
    try:
        from app.config import booking_schema_loader
        reloaders.append(("Booking_schema", booking_schema_loader.reload))
    except Exception as exc:
        results.append({"name": "booking_schema", "ok": False, "error": f"import:{exc}"})
    
    try:
        from app.config import tool_registry_loader
        reloaders.append(("tool_registry", tool_registry_loader.reload))
    except Exception as exc:
        results.append({"name": "tool_registry", "ok": False, "error": f"import:{exc}"})

    try:
        from app.config import agent_policy_loader
        reloaders.append(("agent_policy", agent_policy_loader.reload))
    except Exception as exc:
        results.append({"name": "agent_policy", "ok": False, "error": f"import:{exc}"})

    try:
        from app.config import response_policies_loader
        reloaders.append(("response_policies", response_policies_loader.reload))
    except Exception as exc:
        results.append({"name": "response_policies", "ok": False, "error": f"import:{exc}"})

    try:
        from app.services import dynamic_config as dc
        if hasattr(dc, "reload_all"):
            reloaders.append(("dynamic_config", dc.reload_all))
    except Exception as exc:
        results.append({"name": "dynamic_config", "ok": False, "error": f"import:{exc}"})

    for name, fn in reloaders:
        results.append(_safe_reload(name, fn))
    
    elapsed_ms = round((time.time() - t0) * 1000.0, 2)
    _CONFIG_VERSION += 1
    _LAST_RELOAD_AT = time.time()
    _LAST_RELOAD_RESULT = {
        "version": _CONFIG_VERSION,
        "elapsed_ms": elapsed_ms,
        "results": results,
    }
    logger.info("[admin] reload-config v%d in %.2fms", _CONFIG_VERSION, elapsed_ms)
    return _LAST_RELOAD_RESULT

@router.get("/admin/cofig-version")
async def get_config_version() -> Dict[str, Any]:
    """Public — returns current in-memory config version + last reload metadata."""
    return {
        "config_version": _CONFIG_VERSION,
        "last_reload_at": _LAST_RELOAD_AT,
        "last_reload": _LAST_RELOAD_RESULT or None,
    }