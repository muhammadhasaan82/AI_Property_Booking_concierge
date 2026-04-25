from __future__ import annotations
import concurrent.futures
from torch import multiprocessing
import asyncio
import json
import selectors
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from typing import Any, Dict, List, Optional
import os
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str (_BACKEND))

def _maybe_auto_supabase() -> None:
    auto = ("--auto-supabase" in sys.argv) or (os.getenv("AUTO_SUPABASE") in ("1","true","True"))
    open_studio = ("--open-studio" in sys.argv) or (os.getenv("OPEN_STUDIO") in ("1","true","True"))
    if not auto:
        return
    try:

        subprocess.run(["supabase", "start", "--yes"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:

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
        try:
            from app.services import db_setup as _db
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
    if open_studio:
        try:
            import webbrowser
            webbrowser.open("http://localhost:54323")
        except Exception:
            pass

    if "--auto-supabase" in sys.argv:
        sys.argv.remove("--auto-supabase")
    if "--open-studio" in sys.argv:
        sys.argv.remove("--open-studio")

_maybe_auto_supabase()

from app.services.booking import create_booking, update_booking_status
from app.services.faq import faq_lookup
from app.components.search import property_search
from app.services.adk_runner import run_adk_turn


def _run_async(coro):
    return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop if sys.platform == 'win32' else None)

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


def _load_script_messages(script_path: Optional[str]) -> List[str]:
    if not script_path:
        return []
    path = Path(script_path)
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _resolve_flow_steps(args: argparse.Namespace) -> List[str]:
    steps = list(args.steps or [])
    steps.extend(_load_script_messages(args.script))
    if steps:
        return steps
    if args.demo_memory_fallback:
        return [
            "Show me apartments in New York under $200",
            "I will take option 2",
            "I want to book now",
            "My name is Jane Doe",
            "Email jane@example.com",
            "Phone 5551234567",
            "Check in 2026-05-01",
            "Check out 2026-05-03",
            "2 guests",
            "Yes confirm booking",
        ]
    if not args.demo_booking:
        return []
    return [
        "Show me apartments in New York under $200",
        "Option 1",
        "Book this property",
        "My name is Jane Doe",
        "Email jane@example.com",
        "Phone 5551234567",
        "Check in 2026-05-01",
        "Check out 2026-05-03",
        "2 guests",
        "Yes confirm booking",
    ]
async def setup_multithread_pool():
    """Shifts asyncio to use a multithread pool for parallel computing"""
    workers = multiprocessing.cpu_count() * 5
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    asyncio.get_running_loop().set_default_executor(pool)

async def cmd_chat(args: argparse.Namespace) -> int:
    await setup_multithread_pool()
    session_id = args.session_id or "cli_test_session_001"
    user_id = args.user_id or "cli_user"

    if args.reset_session:
            from app.services.redis_store import clear_session_snapshot
            await clear_session_snapshot(session_id)
            print(f"[Session] Cleared stale for session:{session_id}")

    print("[BOT] AI Concierge Console (interactive). Type 'exit' to quit.")
    def stream_cb(chunk: str):
        print(chunk, end="", flush=True)

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        if user.lower() in ("exit", "quit", ":q"):
            print("Bye!")
            return 0
        print("\nConcierge: ", end="", flush=True)
        async for chunk in run_adk_turn(
            user_id=user_id,
            session_id=session_id,
            message=user,
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print("\n")

async def cmd_say(args: argparse.Namespace) -> int:
    """Send one message through the V2 ADK pipeline."""
    session_id = args.session_id or "cli_say_session"
    user_id = args.user_id or "cli_user"

    print("\nConcierge: ", end="", flush=True)
    async for chunk in run_adk_turn(
        user_id=user_id,
        session_id=session_id,
        message=args.message,
    ):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print("\n")
    return 0


async def cmd_flow(args: argparse.Namespace) -> int:
    await setup_multithread_pool()
    """Run a multi-step conversation through the ADK pipeline."""

    session_id = args.session_id or "cli_flow_session"
    user_id = args.user_id or "cli_user"
    steps = _resolve_flow_steps(args)
    if args.reset_session:
        from app.services.redis_store import clear_session_snapshot
        await clear_session_snapshot(session_id)
        print(f"[Session] Cleared stale for session:{session_id}")
        
    if not steps:
        print("No steps provided. Use --steps, --script, --demo-booking, or --demo-memory-fallback.")
        return 1

    session_id = args.session_id or "cli_flow_session"
    user_id = args.user_id or "cli_user"

    for idx, message in enumerate(steps, 1):
        print(f"\n[Step {idx}/{len(steps)}] You: {message}")
        print("Concierge: ", end="", flush=True)
        async for chunk in run_adk_turn(
            user_id=user_id,
            session_id=session_id,
            message=message,
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print("\n")

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
        sp.add_argument("--session-id", type=str, default=None, help="Reuse a session ID for multi-turn tests")
        sp.add_argument("--user-id", type=str, default=None, help="User ID for the session")
        sp.add_argument("--reset-session", action="store_true", help="Clear session state before running")

    add_common_chat_opts(p)

    sub = p.add_subparsers(dest="cmd")

    sp_chat = sub.add_parser("chat", help="Interactive chat loop (ADK)")
    add_common_chat_opts(sp_chat)
    sp_chat.set_defaults(func=lambda a: _run_async(cmd_chat(a)))

    sp_say = sub.add_parser("say", help="Send one message through the graph")
    add_common_chat_opts(sp_say)
    sp_say.add_argument("message", type=str, help="User message")
    sp_say.set_defaults(func=lambda a: _run_async(cmd_say(a)))

    sp_flow = sub.add_parser("flow", help="Run a multi-step chat flow")
    add_common_chat_opts(sp_flow)
    sp_flow.add_argument("--steps", nargs="*", default=[], help="Messages to send in order")
    sp_flow.add_argument("--script", type=str, default=None, help="Path to a text file with one message per line")
    sp_flow.add_argument("--demo-booking", action="store_true", help="Run a built-in booking demo flow")
    sp_flow.add_argument("--demo-memory-fallback", action="store_true", help="Run a booking flow that relies on memory for property_id")
    # sp_flow.add_argument("--reset-session", action="store_true", help="clear session state before running")
    sp_flow.set_defaults(func=lambda a: _run_async(cmd_flow(a)))
    
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
        return _run_async(cmd_chat(args))
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())

