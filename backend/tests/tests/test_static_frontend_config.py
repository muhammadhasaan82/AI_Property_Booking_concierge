"""Test that static frontend configuration helpers and URLs are correct in app.js."""
from __future__ import annotations

from pathlib import Path


def test_static_frontend_js_config_rules():
    app_js_path = Path("/root/AI_Property_Booking_concierge/frontend_static/app.js")
    assert app_js_path.exists(), f"Could not find {app_js_path}"

    content = app_js_path.read_text(encoding="utf-8")

    assert 'apiUrl("health")' in content or "apiUrl('health')" in content
    assert 'apiUrl("chat/message")' in content or "apiUrl('chat/message')" in content
    assert "docsUrl()" in content

    assert "${apiBaseUrl}/chat/message" not in content
    assert "${apiBaseUrl}/health" not in content
    assert "getApiBaseUrl" not in content

    assert "normalizeBackendOrigin" in content
    assert "/api/v1/" in content
    assert "/docs" in content
