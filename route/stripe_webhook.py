# route/stripe_webhook.py
# Stripe webhook listener for payment event processing.
# Replaces mock payment URL logic with real webhook-driven status updates.

from __future__ import annotations
import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()


@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.
    Processes:
      - checkout.session.completed → update booking to 'confirmed'
      - payment_intent.payment_failed → log failure, keep 'pending'
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # Verify signature if secret is configured
    if STRIPE_WEBHOOK_SECRET:
        try:
            import stripe
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except stripe.error.SignatureVerificationError as e:
            print(f"[STRIPE] Signature verification failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")
        except Exception as e:
            print(f"[STRIPE] Webhook construction error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # No secret configured — parse raw JSON (dev mode)
        import json
        try:
            event = json.loads(payload)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        print("[STRIPE] WARNING: No STRIPE_WEBHOOK_SECRET set. Running in dev mode (no signature verification).")

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    print(f"[STRIPE] Received event: {event_type}")

    if event_type == "checkout.session.completed":
        booking_id = data.get("metadata", {}).get("booking_id") or data.get("client_reference_id")
        if booking_id:
            from services.booking import update_booking_status
            result = await update_booking_status(
                booking_id=booking_id,
                current_status="pending",
                new_status="confirmed",
            )
            print(f"[STRIPE] Booking {booking_id} confirmed: {result}")
        else:
            print(f"[STRIPE] checkout.session.completed but no booking_id in metadata")

    elif event_type == "payment_intent.payment_failed":
        booking_id = data.get("metadata", {}).get("booking_id")
        failure_message = data.get("last_payment_error", {}).get("message", "Unknown error")
        print(f"[STRIPE] Payment failed for booking {booking_id}: {failure_message}")
        # Keep status as 'pending' — do not auto-cancel

    elif event_type == "charge.refunded":
        booking_id = data.get("metadata", {}).get("booking_id")
        if booking_id:
            from services.booking import update_booking_status
            result = await update_booking_status(
                booking_id=booking_id,
                current_status="confirmed",
                new_status="pending",
            )
            print(f"[STRIPE] Booking {booking_id} refunded, status reset: {result}")

    else:
        print(f"[STRIPE] Unhandled event type: {event_type}")

    return JSONResponse(content={"received": True}, status_code=200)
