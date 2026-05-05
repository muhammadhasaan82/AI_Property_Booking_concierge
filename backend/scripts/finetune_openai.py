"""
OpenAI fine-tuning wrapper.

Workflow:
    1. python scripts/export_dpo_dataset.py --mode stfu-router -o stfu_router.jsonl
    2. python scripts/finetune_openai.py upload   stfu_router.jsonl
    3. python scripts/finetune_openai.py create   <FILE_ID> --model gpt-5-nano
    4. python scripts/finetune_openai.py status   <JOB_ID>
    5. ADK_DISPATCHER_MODEL=ft:openai:gpt-5-nano:org:tag:abcd1234 (in .env)

"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
try:
    from openai import OpenAI
except ImportError:
    print("Install: pip install openai>=1.30")
    sys.exit(2)

def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set.")
        sys.exit(2)
    return OpenAI(api_key=key)

def cmd_upload(args: argparse.Namespace) -> None:
    client = _client()
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {p}")
        return 2
    n_lines = sum(1 for _ in p.open("r", encoding="utf-8"))
    print(f"Uploading {p} ({n_lines} examples) …")
    with p.open("rb") as f:
        result = client.files.create(file=f,
        purpose="fine_tune")
    print(json.dumps(result.model_dump(), indent=2,
    default=str))
    print(f"\nFILE_ID: {result.id}")
    return 0

def cmd_create(args: argparse.Namespace) -> int:
    client = _client()
    result = client.fine_tuning.jobs.create(
        training_file=args.file_id,
        model=args.model,
        suffix=args.suffix,
        hyperparameters={"n_epochs": args.epochs} if args.epochs else None,
    )
    print(json.dumps(job.model_dump(), indent=2,
    default=str))
    print(f"\nJOB_ID: {job.id}")
    return 0

def cmd_status(args: argparse.Namespace) -> int:
    client = _client()
    job = client.fine_tuning.jobs.retrieve(args.job_id)
    print(f"id:              {job.id}")
    print(f"status:          {job.status}")
    print(f"model:           {job.model}")
    print(f"fine_tuned_model: {job.fine_tuned_model}")
    print(f"created_at:      {job.created_at}")
    print(f"finished_at:     {getattr(job, 'finished_at', None)}")
    if getattr(job, "error", None):
        print(f"error:           {job.error}")
    if args.events:
        events = client.fine_tuning.jobs.list_events(args.job_id, limit=20)
        for ev in reversed(list(events.data)):
            print(f"- {ev.created_at}: {ev.message}")
    return 0

def cmd_watch(args: argparse.Namespace) -> int:
    """Block until job reaches a terminal state."""
    client = _client()
    print(f"Watching job {args.job_id} …")
    last_status = None
    while True:
        job = client.fine_tuning.jobs.retrieve(args.job_id)
        if job.status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] {job.status}")
            last_status = job.status
        if job.status in ("succeeded", "failed", "cancelled"):
            print(f"final fine_tuned_model: {job.fine_tuned_model}")
            return 0 if job.status == "succeeded" else 1
        time.sleep(args.poll_seconds)
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI fine-tune helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upload", help="upload a JSONL training file")
    p_up.add_argument("file", help="path to .jsonl")
    p_up.set_defaults(func=cmd_upload)

    p_cr = sub.add_parser("create", help="create a fine-tune job")
    p_cr.add_argument("file_id", help="OpenAI file id from `upload`")
    p_cr.add_argument("--model", default="gpt-5-nano")
    p_cr.add_argument("--suffix", default="concierge-router")
    p_cr.add_argument("--epochs", type=int, default=None)
    p_cr.set_defaults(func=cmd_create)

    p_st = sub.add_parser("status", help="check job status")
    p_st.add_argument("job_id")
    p_st.add_argument("--events", action="store_true")
    p_st.set_defaults(func=cmd_status)

    p_wa = sub.add_parser("watch", help="block until job completes")
    p_wa.add_argument("job_id")
    p_wa.add_argument("--poll-seconds", type=int, default=30)
    p_wa.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())