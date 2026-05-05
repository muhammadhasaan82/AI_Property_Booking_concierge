"""
DPO Dataset Export Script —Self-Improvement Feedback Loop.

Queries the telemetry database (SQLite or Supabase) and exports
SUCCESS_PATH / DROP_OFF_PATH trajectory pairs in OpenAI JSONL format
for Direct Preference Optimization fine-tuning of the GPT-5 Nano router.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BACKEND_ROOT))

from dotenv import load_dotenv

_REPO_ROOT = _BACKEND_ROOT 
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

DEFAULT_SQLITE_PATH = str(_BACKEND_ROOT / "dpo_telemetry.db")
SQLITE_PATH = os.getenv("DPO_SQLITE_PATH", DEFAULT_SQLITE_PATH)

DEFAULT_REJECTED = (
    "I'm sorry, I wasn't able to help with that request. "
    "Could you try again or rephrase what you need?"
)

def fetch_from_sqlite(tag: str) -> List[Dict[str, Any]]:
    """Fetch trajectories from the local SQLite DB."""
    if not Path(SQLITE_PATH).exists():
        print(f"[WARN] SQLite DB not found at {SQLITE_PATH}")
        return []

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT session_id, user_id, user_message, tool_calls,
               final_reply, booking_id, turn_count, latency_ms, created_at,
               cognitive_context, understanding_frame_json, policy_override_json
        FROM dpo_trajectories
        WHERE trajectory_tag = ?
        ORDER BY created_at DESC
        """,
        (tag,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows



async def fetch_from_supabase(tag: str) -> List[Dict[str, Any]]:
    """Fetch trajectories from the Supabase DB."""
    try:
        from app.services import db_client
        rows = await db_client.fetch_all(
            """
            SELECT session_id, user_id, user_message, tool_calls,
                   final_reply, booking_id, turn_count, latency_ms, created_at
            FROM public.dpo_trajectories
            WHERE trajectory_tag = %s
            ORDER BY created_at DESC
            """,
            (tag,),
        )
        return rows or []
    except Exception as e:
        print(f"[WARN] Supabase fetch failed: {e}")
        return []


def fetch_trajectories(
    tag: str,
    source: str = "auto",
) -> List[Dict[str, Any]]:
    """Fetch trajectories from the specified source."""
    if source == "sqlite":
        return fetch_from_sqlite(tag)
    elif source == "supabase":
        return asyncio.run(fetch_from_supabase(tag))
    else:
        rows = fetch_from_sqlite(tag)
        if rows:
            return rows
        return asyncio.run(fetch_from_supabase(tag))

def _normalize_message(msg: str) -> str:
    """Normalize a user message for fuzzy matching."""
    return " ".join((msg or "").lower().split())


def build_dpo_pairs(
    success_rows: List[Dict[str, Any]],
    dropoff_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Build DPO preference pairs.

    Strategy:
    1. Try to match SUCCESS and DROP_OFF trajectories by similar user_message.
    2. For unmatched successes, pair with a synthetic rejected response.
    3. For unmatched dropoffs, pair with a synthetic chosen response.

    Returns:
        (pairs, stats) where pairs = [{"prompt", "chosen", "rejected"}, ...]
    """
    pairs: List[Dict[str, str]] = []
    stats = {"natural_pairs": 0, "synthetic_chosen": 0, "synthetic_rejected": 0}

    dropoff_index: Dict[str, List[Dict[str, Any]]] = {}
    for row in dropoff_rows:
        key = _normalize_message(row.get("user_message", ""))
        if key:
            dropoff_index.setdefault(key, []).append(row)

    matched_dropoff_keys: set = set()

    for s_row in success_rows:
        s_msg = _normalize_message(s_row.get("user_message", ""))
        s_reply = (s_row.get("final_reply") or "").strip()
        if not s_msg or not s_reply:
            continue

        if s_msg in dropoff_index and dropoff_index[s_msg]:
            d_row = dropoff_index[s_msg].pop(0)
            d_reply = (d_row.get("final_reply") or "").strip() or DEFAULT_REJECTED
            pairs.append({
                "prompt": s_row.get("user_message", ""),
                "chosen": s_reply,
                "rejected": d_reply,
            })
            matched_dropoff_keys.add(s_msg)
            stats["natural_pairs"] += 1
        else:
            pairs.append({
                "prompt": s_row.get("user_message", ""),
                "chosen": s_reply,
                "rejected": DEFAULT_REJECTED,
            })
            stats["synthetic_rejected"] += 1

    for key, d_rows in dropoff_index.items():
        for d_row in d_rows:
            d_msg = (d_row.get("user_message") or "").strip()
            d_reply = (d_row.get("final_reply") or "").strip()
            if not d_msg or not d_reply:
                continue
            pairs.append({
                "prompt": d_msg,
                "chosen": (
                    "I'd be happy to help you with that! Let me find the best "
                    "options for you right away."
                ),
                "rejected": d_reply,
            })
            stats["synthetic_chosen"] += 1

    return pairs, stats

_LOW_QUALITY_PHRASES = (
    "i'm not sure",
    "i don't know",
    "something went wrong",
    "an error occured",
)
_MIN_REPLY_LEN = 12

def is_low_quality(reply: str) -> bool:
    if not reply or len(reply.strip()) < _MIN_REPLY_LEN:
        return True
    low = reply.lower()
    return any(phrase in low for phrase in _LOW_QUALITY_PHRASES)

def deduplicate_by_message(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:    
    seen: set = set()
    out: List[Dict[str,Any]] = []
    for r in rows:
        key = _normalize_message(r.get("user_message", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def parse_frame(row: Dict[str, Any]) -> DIct[str, Any]:
    raw = row.get("understanding_frame_json")
    if not raw:
        return{}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return {}

def parse_override(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("policy_override_json")
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return{}

def parse_tool_calls(row: Dict[str, Any]) -> List[Dict[str,
Any]]:
    raw = row.get("tool_calls")
    if not raw:
        return[]
    try:
        return json.loads(raw) if isinstance(raw, str) else list(raw)
    except Exception:
        return[]

def balance_by_intent(rows: List[Dict[str, Any]],
cap_per_intent: int = 200) -> List[Dict[str, Any]]:
    """Cap rows per primary_intent to pervent class imbalance
    dominating training."""
    by_intent: Dict[str, List[Dict[str,Any]]] = {}
    for r in rows:
        frame = parse_frame(r)
        intent = frame.get("primary_intent", "unknown")
        by_intent.setdefault(intent, []).append(r)
    out: List[Dict[str, Any]] = []
    for intent, rs in by_intent.items():
        out.extend(rs[:cap_per_intent])
    return out

def build_stfu_understanding_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """STFU for the understanding_agent: prompt → UnderstandingFrame JSON."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        frame = parse_frame(r)
        if not frame or not frame.get("primary_intent"):
            continue
        msg = r.get("user_message", "").strip()
        if not msg:
            continue
        out.append({
            "message": [
                {"role":"system",
                 "content": "Emit an UnderstandingFrame JSON for the user message."},
                {"role": "user", "content": msg},
                {"role": "assistant", "content": json.dumps
                 (frame, ensure_ascii=False)},
            
            ]
        })
    return out

def build_stfu_router_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """STFU for the triage_router: prompt → tool call.
    
    Quality gates:
      - Skip rows with policy_override_json (policy disagreed → noisy training data)
      - Require at least one tool call
      - Require trajectory_tag = SUCCESS_PATH at fetch time (caller's job)
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("policy_override_json"):
            continue
        calls = parse_tool_calls(r)
        if not calls:
            continue
        msg = r.get("user_message", "").strip()
        if not msg:
            continue
        first = calls[0]
        tool_name = first.get("tool")
        if not tool_name:
            continue
        out.append({
            "messages": [
                {"role": "system",
                "content": "Call exactly one tool with the best-guess arguments."},
                {"role": "user", "content": msg},
                {"role": "assistant",
                "content": "",
                "tool_calls":  [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(first.get
                        ("params") or {},
                        ensure_ascii=False),
                    },
                }]},
            ]
        })
    return out

def export_jsonl(pairs: List[Dict[str, str]], output_path: str) -> None:
    """Write DPO pairs to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DPO/STFU training data from telemetry."
    )
    parser.add_argument(
        "--output", "-o",
        default="dpo_dataset.jsonl",
    )
    parser.add_argument(
        "--source", "-s",
        choices=["auto", "sqlite", "supabase"],
        default="auto",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["dpo-voice", "stfu-router", "stfu-understanding"],
        default="dpo-voice",
        help=(
            "dpo-voice: voice perference pairs (existing behaviour)\n"
            "stfu-router: tool-call training data for the dispatcher\n"
            "stfu-understanding: structured-frame training for understanding_agent"
        ),
    )
    parser.add_argument("--cap-per-intent", type=int, default=200,
                        help="Cap rows per primary_intent (stfu modes)")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="skip deduplication by user message")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Skip the low-quality filter")
    args = parser.parse_args()

    print(f"[Export] Mode = {args.mode} | source = {args.source}")
    print(f"[Export] SQLite path: {SQLITE_PATH}")
    print()

    if args.mode == "dpo-voice":
        success_rows = fetch_trajectories("SUCCESS_PATH", args.source)
        dropoff_rows = fetch_trajectories("DROP_OFF_PATH", args.source)
        print(f"[Export] success={len(success_rows)}, dropoff={len(dropoff_rows)}")

        if not args.no_quality_filter:
            success_rows = [r for r in success_rows if not is_low_quality(r.get("final_reply", ""))]
        if not args.no_dedupe:  
            success_rows = deduplicate_by_message(success_rows)
            dropoff_rows = deduplicate_by_message(dropoff_rows)

        pairs, stats = build_dpo_pairs(success_rows, dropoff_rows)
        if not pairs:
            print("[Export] No pairs built.")
            sys.exit(0)
        export_jsonl(pairs, args.output)
        print(f"[Export] Wrote {len(pairs)} preference pairs → {args.output}")
        print(f" natural_pairs={stats['natural_pairs']})"
              f" synthetic_chosen={stats['synthetic_chosen']}"
              f" synthetic_rejected={stats['synthetic_rejected']}")
    
    elif args.mode == "stfu-router":
        rows = fetch_trajectories("SUCCESS_PATH", args.source)
        print(f"[Export] success rows={len(rows)}")
        if not args.no_dedupe:
            rows = deduplicate_by_message(rows)
        rows = balance_by_intent(rows, cap_per_intent=args.cap_per_intent)
        examples = build_stfu_router_pairs(rows)
        export_jsonl(examples, args.output)
        print(f"[Export] wrote {len(examples)} STFU router examples → {args.output}")
    
    elif args.mode == "stfu-understanding":
        rows = fetch_trajectories("SUCCESS_PATH", args.source)
        rows += fetch_trajectories("IN_PROGRESS", args.source)
        print(f"[Export] rows (success+in_progress)={len(rows)}")
        if not args.no_dedupe:
            rows = deduplicate_by_message(rows)
        rows = balance_by_intent(rows, cap_per_intent=args.cap_per_intent)
        examples = build_stfu_understanding_pairs(rows)
        export_jsonl(examples, args.output)
        print(f"[Export] wrote {len(examples)} STFU understanding examples → {args.output}")

    print("[Export] done.")
    success_rows = fetch_trajectories("SUCCESS_PATH", args.source)
    print(f"  Found: {len(success_rows)}")

    print("[DPO Export] Fetching DROP_OFF_PATH trajectories...")
    dropoff_rows = fetch_trajectories("DROP_OFF_PATH", args.source)
    print(f"  Found: {len(dropoff_rows)}")
    print()

    if not success_rows and not dropoff_rows:
        print("[DPO Export] No trajectory data found. Run the chatbot first to generate telemetry.")
        sys.exit(0)
    pairs, stats = build_dpo_pairs(success_rows, dropoff_rows)

    if not pairs:
        print("[DPO Export] No valid pairs could be constructed.")
        sys.exit(0)

    export_jsonl(pairs, args.output)

    print(f"[DPO Export] Exported {len(pairs)} preference pairs to {args.output}")
    print(f"  Natural pairs:      {stats['natural_pairs']}")
    print(f"  Synthetic rejected:  {stats['synthetic_rejected']}")
    print(f"  Synthetic chosen:    {stats['synthetic_chosen']}")
    print()
    print("[DPO Export] Ready for fine-tuning with OpenAI DPO API.")


if __name__ == "__main__":
    main()
