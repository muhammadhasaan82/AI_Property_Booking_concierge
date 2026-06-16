from __future__ import annotations

from evaluation.eval_dataset import load_dataset
from evaluation.eval_runner import run_evaluation_sync
from evaluation.report_card import build_report_card, render_markdown_report, render_terminal_report
from scripts.check_app_file_sizes import main as file_size_main


def test_report_card_contains_summary_and_ci_decision():
    dataset = load_dataset("evaluation/golden_set.yaml")
    samples = [sample for sample in dataset.samples if sample.id == "robustness_irrelevant_query_01"]
    results = run_evaluation_sync(dataset, samples)
    report = build_report_card(results, fail_under=0.85)
    assert report["total_cases"] == 1
    assert report["passed_cases"] == 1
    assert report["ci"]["decision"] == "PASS"
    assert report["latency_ms"]["max"] >= 0


def test_report_card_renders_terminal_and_markdown():
    dataset = load_dataset("evaluation/golden_set.yaml")
    samples = [sample for sample in dataset.samples if sample.id == "robustness_irrelevant_query_01"]
    results = run_evaluation_sync(dataset, samples)
    report = build_report_card(results, fail_under=0.85)
    assert "AI Report Card" in render_terminal_report(report)
    assert "# AI Report Card" in render_markdown_report(report)


def test_app_file_size_check_still_passes():
    assert file_size_main() == 0
