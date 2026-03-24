#!/usr/bin/env python3
"""
DPO Dataset Export Script — Phase 3 Self-Improvement Feedback Loop.

Queries the telemetry database (SQLite or Supabase) and exports
SUCCESS_PATH / DROP_OFF_PATH trajectory pairs in OpenAI JSONL format
for Direct Preference Optimization fine-tuning of the GPT-5 Nano router.

Usage:
    python scripts/export_dpo_dataset.py --output dpo_dataset.jsonl
    python scripts/export_dpo_dataset.py --output dpo_dataset.jsonl --source sqlite
    python scripts/export_dpo_dataset.py --output dpo_dataset.jsonl --source supabase
    python scripts/export_dpo_dataset.py --output dpo_dataset.jsonl --source auto
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

# Ensure backend is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_BACKEND_ROOT))

from dotenv import load_dotenv

_REPO_ROOT = _BACKEND_ROOT  # scripts/ is now at root level
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# Default SQLite path
DEFAULT_SQLITE_PATH = str(_BACKEND_ROOT / "dpo_telemetry.db")
SQLITE_PATH = os.getenv("DPO_SQLITE_PATH", DEFAULT_SQLITE_PATH)

# Placeholder for rejected responses when no natural pair exists
DEFAULT_REJECTED = (
    "I'm sorry, I wasn't able to help with that request. "
    "Could you try again or rephrase what you need?"
)


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

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
               final_reply, booking_id, turn_count, latency_ms, created_at
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
        # Auto: try SQLite first, fall back to Supabase
        rows = fetch_from_sqlite(tag)
        if rows:
            return rows
        return asyncio.run(fetch_from_supabase(tag))


# ---------------------------------------------------------------------------
# Pairing logic
# ---------------------------------------------------------------------------

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

    # Index dropoffs by normalized message for matching
    dropoff_index: Dict[str, List[Dict[str, Any]]] = {}
    for row in dropoff_rows:
        key = _normalize_message(row.get("user_message", ""))
        if key:
            dropoff_index.setdefault(key, []).append(row)

    matched_dropoff_keys: set = set()

    # Match successes to dropoffs
    for s_row in success_rows:
        s_msg = _normalize_message(s_row.get("user_message", ""))
        s_reply = (s_row.get("final_reply") or "").strip()
        if not s_msg or not s_reply:
            continue

        # Try exact message match
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
            # Synthetic: use default rejected
            pairs.append({
                "prompt": s_row.get("user_message", ""),
                "chosen": s_reply,
                "rejected": DEFAULT_REJECTED,
            })
            stats["synthetic_rejected"] += 1

    # Unmatched dropoffs get synthetic chosen
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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_jsonl(pairs: List[Dict[str, str]], output_path: str) -> None:
    """Write DPO pairs to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DPO preference pairs from telemetry data."
    )
    parser.add_argument(
        "--output", "-o",
        default="dpo_dataset.jsonl",
        help="Output JSONL file path (default: dpo_dataset.jsonl)",
    )
    parser.add_argument(
        "--source", "-s",
        choices=["auto", "sqlite", "supabase"],
        default="auto",
        help="Data source (default: auto — tries SQLite then Supabase)",
    )
    args = parser.parse_args()

    print(f"[DPO Export] Source: {args.source}")
    print(f"[DPO Export] SQLite path: {SQLITE_PATH}")
    print()

    # Fetch trajectories
    print("[DPO Export] Fetching SUCCESS_PATH trajectories...")
    success_rows = fetch_trajectories("SUCCESS_PATH", args.source)
    print(f"  Found: {len(success_rows)}")

    print("[DPO Export] Fetching DROP_OFF_PATH trajectories...")
    dropoff_rows = fetch_trajectories("DROP_OFF_PATH", args.source)
    print(f"  Found: {len(dropoff_rows)}")
    print()

    if not success_rows and not dropoff_rows:
        print("[DPO Export] No trajectory data found. Run the chatbot first to generate telemetry.")
        sys.exit(0)

    # Build pairs
    pairs, stats = build_dpo_pairs(success_rows, dropoff_rows)

    if not pairs:
        print("[DPO Export] No valid pairs could be constructed.")
        sys.exit(0)

    # Export
    export_jsonl(pairs, args.output)

    print(f"[DPO Export] Exported {len(pairs)} preference pairs to {args.output}")
    print(f"  Natural pairs:      {stats['natural_pairs']}")
    print(f"  Synthetic rejected:  {stats['synthetic_rejected']}")
    print(f"  Synthetic chosen:    {stats['synthetic_chosen']}")
    print()
    print("[DPO Export] Ready for fine-tuning with OpenAI DPO API.")


if __name__ == "__main__":
    main()
