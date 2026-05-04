# Fine-Tuning Playbook

End-to-end runbook for closing the self-improvement loop:
collect → evaluate → clean → train → swap → re-evaluate.

## 0. Prerequisites

- Telemetry must have at least ~500 SUCCESS_PATH rows. Verify:
  ```bash
  sqlite3 backend/dpo_telemetry.db \
    "SELECT trajectory_tag, COUNT(*) FROM dpo_trajectories GROUP BY trajectory_tag;"
  ```
- `OPENAI_API_KEY` in env, with fine-tuning quota.

## 1. Baseline eval

```bash
cd backend
python evaluation/run_eval_suite.py --json --out evaluation/eval_results/baseline.json
```

Record the metrics — this is your "before" measurement.

## 2. Export training data

Pick one target. For the dispatcher (most useful):

```bash
python scripts/export_dpo_dataset.py --mode sft-router \
    --output evaluation/eval_results/sft_router.jsonl \
    --cap-per-intent 200
```

For the understanding agent:

```bash
python scripts/export_dpo_dataset.py --mode sft-understanding \
    --output evaluation/eval_results/sft_understanding.jsonl
```

For the voice agent (DPO):

```bash
python scripts/export_dpo_dataset.py --mode dpo-voice \
    --output evaluation/eval_results/dpo_voice.jsonl
```

Inspect a few examples:

```bash
head -3 evaluation/eval_results/sft_router.jsonl | python -m json.tool
```

## 3. Train

```bash
# Upload
python scripts/finetune_openai.py upload evaluation/eval_results/sft_router.jsonl
# → records FILE_ID

# Kick off
python scripts/finetune_openai.py create file-abc123 \
    --model gpt-5-nano --suffix concierge-router

# Watch (blocks until done)
python scripts/finetune_openai.py watch ftjob-xyz789
```

When complete, the script prints `final fine_tuned_model = ft:openai:gpt-5-nano:org:concierge-router:abcd1234`.

## 4. Swap & A/B

Edit [backend/.env](file:///c:/Users/ASUS/Desktop/Hotel%20booking/backend/.env:0:0-0:0):

```bash
ADK_DISPATCHER_MODEL=ft:openai:gpt-5-nano:org:concierge-router:abcd1234
```

Run the comparison:

```bash
python evaluation/eval_compare_models.py \
    --baseline openai/gpt-5-nano \
    --candidate ft:openai:gpt-5-nano:org:concierge-router:abcd1234
```

Look for positive deltas on `tool_selection_accuracy`, `arg_extraction_accuracy`, and `frame_intent_accuracy`. Negative deltas on `policy_override_rate` mean the model now agrees with the policy router more often (good).

## 5. Promote or rollback

- If the candidate beats the baseline on every key metric **and** error rate didn't increase → keep it.
- Otherwise revert `ADK_DISPATCHER_MODEL`. The change is one env var.

## Alternatives

- **Together AI**: same JSONL format works. Endpoint is `https://api.together.xyz/v1/fine-tunes`. See their docs.
- **Local TRL (HuggingFace)**: convert JSONL with `trl.SFTTrainer`. Useful for offline labs but slower.

## Tips

- Re-export and re-train monthly — the model improves as you accumulate more telemetry.
- Keep a per-tag breakdown across runs: regression on a specific tag (e.g. `selection`) is more informative than aggregate accuracy.
- Don't fine-tune on raw telemetry without the policy filter (`policy_override_json IS NULL`) — that bakes in the model's old mistakes.
