from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from evaluation.eval_dataset import filter_samples, load_dataset
from evaluation.eval_runner import run_evaluation_sync
from evaluation.integrations.langfuse_eval import publish_results
from evaluation.judge import evaluate_case_with_judge


DATASET = Path("evaluation/golden_set.yaml")


def test_eval_runner_single_search_smoke():
    dataset = load_dataset(DATASET)
    samples = [sample for sample in dataset.samples if sample.id == "search_nyc_apartment_alias_01"]
    results = run_evaluation_sync(dataset, samples)
    assert len(results) == 1
    assert results[0].passed
    assert results[0].turns[0].response


def test_tags_filter_cases():
    dataset = load_dataset(DATASET)
    filtered = filter_samples(dataset.samples, "search")
    assert filtered
    assert all("search" in sample.tags for sample in filtered)


def test_cli_json_outputs_valid_json_for_tagged_run():
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    completed = subprocess.run(
        [
            sys.executable,
            "evaluation/v2_eval.py",
            "--dataset",
            "evaluation/golden_set.yaml",
            "--tags",
            "irrelevant",
            "--json",
            "--fail-under",
            "0.0",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["report_card"]["total_cases"] == 1
    assert payload["results"][0]["id"] == "robustness_irrelevant_query_01"


def test_llm_judge_skipped_when_disabled():
    dataset = load_dataset(DATASET)
    result = run_evaluation_sync(dataset, [dataset.samples[0]])[0]
    judged = evaluate_case_with_judge(result, mode="none")
    assert judged["judge_skipped"] is True


def test_langfuse_integration_skipped_when_disabled(monkeypatch):
    monkeypatch.delenv("EVAL_LANGFUSE_ENABLED", raising=False)
    skipped = publish_results(run_id="test", results=[])
    assert skipped["skipped"] is True
