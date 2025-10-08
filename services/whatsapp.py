# services/whatsapp.py
# Async WhatsApp sender stub (replace with your actual provider, e.g., Meta API or Twilio).

from __future__ import annotations
import os
from typing import Dict, Any
import httpx

_WA_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
_WA_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")

async def send_payment_link_async(booking_id: str, phone: str, url: str, preview: bool = True) -> Dict[str, Any]:
    """
    Sends a simple WhatsApp text message with the payment link.
    For production, you may switch to a message template.
    """
    if not (_WA_TOKEN and _WA_PHONE_ID):
        return {"ok": False, "error": "WhatsApp env not configured"}

    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {
            "preview_url": preview,
            "body": f"Your payment link for booking {booking_id}: {url}"
        }
    }
    headers = {"Authorization": f"Bearer {_WA_TOKEN}"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"https://graph.facebook.com/v17.0/{_WA_PHONE_ID}/messages",
                              headers=headers, json=payload)
    if r.status_code in (200, 201):
        return {"ok": True, "delivered": True, "booking_id": booking_id, "to": phone, "link": url}
    return {"ok": False, "error": r.text}
