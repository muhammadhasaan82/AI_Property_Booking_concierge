#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional
import os
import subprocess

# ------------------------------------------------------------
# Optional: auto-start Supabase and export env before service imports
# Usage: add --auto-supabase to your command, or set AUTO_SUPABASE=1
# ------------------------------------------------------------
def _maybe_auto_supabase() -> None:
    auto = ("--auto-supabase" in sys.argv) or (os.getenv("AUTO_SUPABASE") in ("1","true","True"))
    open_studio = ("--open-studio" in sys.argv) or (os.getenv("OPEN_STUDIO") in ("1","true","True"))
    if not auto:
        return
    try:
        # Start services (idempotent if already running)
        subprocess.run(["supabase", "start", "--yes"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        # Read env values for this project
        res = subprocess.run(["supabase", "status", "-o", "env"], check=False, capture_output=True, text=True)
        for line in (res.stdout or "").splitlines():
            if line.startswith("API_URL="):
                os.environ["SUPABASE_URL"] = line.split("=", 1)[1].strip()
            elif line.startswith("SERVICE_ROLE_KEY="):
                key = line.split("=", 1)[1].strip()
                os.environ["SUPABASE_KEY"] = key
                os.environ["SUPABASE_SERVICE_ROLE_KEY"] = key
            elif line.startswith("ANON_KEY="):
                os.environ["SUPABASE_ANON_KEY"] = line.split("=", 1)[1].strip()
        print("[Supabase] Studio: http://localhost:54323", flush=True)
        print("[Supabase] Tables: public.chat_history, public.successful_bookings", flush=True)
        # Auto-init schema to ensure tables exist
        try:
            from services import db_setup as _db
            _db.init_schema(None)
            snap = _db.verify(None)
            print(
                f"[Supabase] Verify: users={snap.get('users')} bookings={snap.get('bookings')} "
                f"chat_history={snap.get('chat_history')} successful_bookings={snap.get('successful_bookings')}",
                flush=True,
            )
        except Exception as _e:
            print(f"[Supabase] Schema init skipped: {_e}", flush=True)
    except Exception:
        pass
    # Optionally open Studio in a browser
    if open_studio:
        try:
            import webbrowser
            webbrowser.open("http://localhost:54323")
        except Exception:
            pass
    # Remove the flag before argparse
    if "--auto-supabase" in sys.argv:
        sys.argv.remove("--auto-supabase")
    if "--open-studio" in sys.argv:
        sys.argv.remove("--open-studio")

_maybe_auto_supabase()

from services.graph import run_chat_graph
from services.booking import create_booking, update_booking_status
from services.faq import faq_lookup
from services.search import property_search

def _parse_kv_list(values: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not values: return out
    for item in values:
        if "=" not in item: continue
        k, v = item.split("=", 1)
        v = v.strip()
        try:
            if not v: out[k]=v
            elif v[0] in "{[": out[k]=json.loads(v)
            elif v.lower() in ("true","false"): out[k]=(v.lower()=="true")
            elif "." in v: out[k]=float(v)
            else: out[k]=int(v)
        except Exception:
            out[k]=v
    return out

def _pretty(o: Any) -> str: return json.dumps(o, indent=2, ensure_ascii=False)

async def cmd_chat(args: argparse.Namespace) -> int:
    print("[BOT] AI Concierge Console (interactive). Type 'exit' to quit.")
    # Streaming callback for CLI (default ON)
    def stream_cb(chunk: str):
        print(chunk, end="", flush=True)

    session_filters: Dict[str, Any] = {
        "budget": args.budget,
        "beds": args.beds,
        "location": args.city,
        "amenities": args.amenities or [],
        "locale": args.locale,
        "stream": not args.no_stream,         # default True
        "stream_callback": stream_cb if not args.no_stream else None,
    }
    session_booking = _parse_kv_list(args.booking_args)
    session_status = _parse_kv_list(args.status_args)
    session_payment = _parse_kv_list(args.payment_args)

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        if user.lower() in ("exit", "quit", ":q"):
            print("Bye!")
            return 0

        overrides = _parse_kv_list(args.with_kv)
        current_filters = {**session_filters, **overrides}

        result = await run_chat_graph(
            message=user,
            filters=current_filters,
            booking_args=session_booking,
            status_args=session_status,
            payment_args=session_payment,
        )

        # Persist states returned by graph natively into local dicts
        if result.get("filters"):
            session_filters = result["filters"]
        if result.get("booking_args"):
            session_booking = result["booking_args"]
        if result.get("status_args"):
            session_status = result["status_args"]
        if result.get("payment_args"):
            session_payment = result["payment_args"]

        print(f"\n[BOT] ({result.get('intent','?')}): {result.get('reply','(no reply)')}")

async def cmd_say(args: argparse.Namespace) -> int:
    def stream_cb(chunk: str):
        print(chunk, end="", flush=True)

    result = await run_chat_graph(
        message=args.message,
        filters={
            "budget": args.budget,
            "beds": args.beds,
            "location": args.city,
            "amenities": args.amenities or [],
            "locale": args.locale,
            "stream": not args.no_stream,      # default True
            "stream_callback": stream_cb if not args.no_stream else None,
        },
        booking_args=_parse_kv_list(args.booking_args),
        status_args=_parse_kv_list(args.status_args),
        payment_args=_parse_kv_list(args.payment_args),
    )
    print(f"\nBOT ({result.get('intent','?')}): {result.get('reply','(no reply)')}")
    return 0

def cmd_search(args: argparse.Namespace) -> int:
    results = property_search(
        query_text=args.query or "",
        budget=args.budget,
        amenities=args.amenities or [],
        location=args.city,
        beds=args.beds,
    )
    print(_pretty(results)); return 0

def cmd_faq(args: argparse.Namespace) -> int:
    ans = faq_lookup(args.question)
    print(ans or "No exact FAQ match."); return 0

def cmd_booking_create(args: argparse.Namespace) -> int:
    payload = {
        "user_id": args.user_id,
        "property_id": args.property_id,
        "check_in": args.check_in,
        "check_out": args.check_out,
        "guests": args.guests,
        "phone": args.phone,
    }
    out = create_booking(payload)
    print(_pretty(out)); return 0 if out.get("ok") else 1

def cmd_booking_update(args: argparse.Namespace) -> int:
    out = update_booking_status(
        booking_id=args.booking_id,
        current_status=args.current_status,
        new_status=args.new_status,
    )
    print(_pretty(out)); return 0 if out.get("ok") else 1

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chatbot", description="Terminal runner for the AgenticRAG property assistant.")

    def add_common_chat_opts(sp):
        sp.add_argument("--budget", type=float, default=None, help="Max nightly budget")
        sp.add_argument("--beds", type=int, default=None, help="Minimum bedrooms")
        sp.add_argument("--city", type=str, default=None, help="City filter")
        sp.add_argument("--amenities", nargs="*", default=None, help="Amenity filters (space-separated)")
        sp.add_argument("--locale", type=str, default="en", help="Reply language (default: en)")
        sp.add_argument("--booking-args", nargs="*", default=[], help="Extra booking args as key=value")
        sp.add_argument("--status-args", nargs="*", default=[], help="Extra status args as key=value")
        sp.add_argument("--payment-args", nargs="*", default=[], help="Extra payment args as key=value")
        sp.add_argument("--with", dest="with_kv", nargs="*", default=[], help="Per-message filter overrides key=value")
        sp.add_argument("--no-stream", action="store_true", help="Disable streaming (default: stream enabled)")
        sp.add_argument("--auto-supabase", action="store_true", help="Auto-start Supabase and export env for this session")

    # root options (so running without subcommand works)
    add_common_chat_opts(p)

    sub = p.add_subparsers(dest="cmd")

    sp_chat = sub.add_parser("chat", help="Interactive chat loop (LangGraph)")
    add_common_chat_opts(sp_chat)
    sp_chat.set_defaults(func=lambda a: asyncio.run(cmd_chat(a)))

    sp_say = sub.add_parser("say", help="Send one message through the graph")
    add_common_chat_opts(sp_say)
    sp_say.add_argument("message", type=str, help="User message")
    sp_say.set_defaults(func=lambda a: asyncio.run(cmd_say(a)))

    sp_search = sub.add_parser("search", help="Direct property search (debug)")
    sp_search.add_argument("--query", type=str, default="", help="Free-text query")
    sp_search.add_argument("--budget", type=float, default=None)
    sp_search.add_argument("--beds", type=int, default=None)
    sp_search.add_argument("--city", type=str, default=None)
    sp_search.add_argument("--amenities", nargs="*", default=None)
    sp_search.set_defaults(func=cmd_search)

    sp_faq = sub.add_parser("faq", help="Query the FAQs table")
    sp_faq.add_argument("question", type=str)
    sp_faq.set_defaults(func=cmd_faq)

    sp_bc = sub.add_parser("booking-create", help="Create a booking (DB + Stripe session/link)")
    sp_bc.add_argument("--user-id", required=True)
    sp_bc.add_argument("--property-id", required=True)
    sp_bc.add_argument("--check-in", required=True, help="YYYY-MM-DD")
    sp_bc.add_argument("--check-out", required=True, help="YYYY-MM-DD")
    sp_bc.add_argument("--guests", type=int, default=1)
    sp_bc.add_argument("--phone", type=str, default=None)
    sp_bc.set_defaults(func=cmd_booking_create)

    sp_bu = sub.add_parser("booking-update", help="Update booking status")
    sp_bu.add_argument("--booking-id", required=True)
    sp_bu.add_argument("--current-status", required=True, choices=["pending","confirmed","checked_in","checked_out"])
    sp_bu.add_argument("--new-status", required=True, choices=["pending","confirmed","checked_in","checked_out"])
    sp_bu.set_defaults(func=cmd_booking_update)

    return p

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) is None:
        return asyncio.run(cmd_chat(args))
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
