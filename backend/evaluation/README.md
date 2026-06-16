# AI Report Card Evals

The v2 eval system runs the AI Property Booking Concierge against
`evaluation/golden_set.yaml`. The YAML file is the source of truth for golden
samples, fixtures, expected responses, expected state, and scoring weights.

## Local Run

```bash
uv run env PYTHONPATH=. python evaluation/v2_eval.py
```

JSON output:

```bash
uv run env PYTHONPATH=. python evaluation/v2_eval.py --json
```

Write the latest report card:

```bash
uv run env PYTHONPATH=. python evaluation/v2_eval.py \
  --dataset evaluation/golden_set.yaml \
  --json \
  --out evaluation/eval_results/latest.json \
  --report-card
```

## Run By Tag

```bash
uv run env PYTHONPATH=. python evaluation/v2_eval.py --tags search,booking
```

Tags are matched as an OR filter.

## CI Command

The deterministic CI path needs no external API keys:

```bash
uv run env PYTHONPATH=. python evaluation/v2_eval.py \
  --dataset evaluation/golden_set.yaml \
  --json \
  --out evaluation/eval_results/latest.json \
  --ci \
  --fail-under 0.85
```

The eval harness runs through the local app pipeline with fixture property data,
in-memory session state, and stubbed booking persistence. Booking writes are
dry-run by default.

## LLM Judge

The LLM judge is disabled by default and should only be used for subjective
dimensions such as helpfulness, policy faithfulness, refusal quality, and
unsupported-city correctness.

```bash
EVAL_ENABLE_LLM_JUDGE=1 OPENAI_API_KEY=... \
uv run env PYTHONPATH=. python evaluation/v2_eval.py --judge llm
```

If the judge is unavailable it is skipped unless `--require-judge` is set.

## Langfuse

Langfuse publishing is optional and disabled unless explicitly enabled:

```bash
EVAL_LANGFUSE_ENABLED=1 \
uv run env PYTHONPATH=. python evaluation/v2_eval.py --json
```

Published payloads include run id, sample id, tags, responses, deterministic
score, judge score when present, latency, pass/fail, and failure reasons. Emails,
phone numbers, and booking IDs are redacted before publishing. Integration
errors do not fail the eval unless `--strict-integrations` is provided.

## Braintrust And Promptfoo Exports

These adapters only export files and do not require Braintrust or promptfoo as
dependencies.

```bash
uv run env PYTHONPATH=. python evaluation/v2_eval.py \
  --export-braintrust evaluation/exports/braintrust_cases.jsonl

uv run env PYTHONPATH=. python evaluation/v2_eval.py \
  --export-promptfoo evaluation/exports/promptfoo.yaml
```

## Reading The Report Card

The report card shows total cases, pass rate, average deterministic score,
optional LLM judge score, per-category metrics, latency summary, failures, top
regressions, slowest cases, and the CI decision.

Failures are intentionally explicit. They call out missing required text,
forbidden text, expected tool mismatches, argument mismatches, missing state
assertions, exceptions, and unsafe final-booking claims.
