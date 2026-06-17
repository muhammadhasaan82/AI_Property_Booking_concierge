from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_set.yaml"


@dataclass
class Expected:
    tool: str | None = None
    intent: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    response_contains: list[str] = field(default_factory=list)
    response_not_contains: list[str] = field(default_factory=list)
    state_assertions: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalTurn:
    user: str
    expected: Expected = field(default_factory=Expected)


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(default_factory=dict)
    pass_threshold: float = 0.8
    strict: bool = False


@dataclass
class EvalSample:
    id: str
    type: str
    tags: list[str]
    expected: Expected
    scoring: ScoringConfig
    prompt: str | None = None
    turns: list[EvalTurn] = field(default_factory=list)
    initial_soft_state: dict[str, Any] = field(default_factory=dict)
    allow_final_booking: bool = False


@dataclass
class EvalDataset:
    version: str
    samples: list[EvalSample]
    defaults: dict[str, Any] = field(default_factory=dict)
    fixtures: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


class DatasetValidationError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expected_from(raw: dict[str, Any] | None, default: dict[str, Any] | None) -> Expected:
    data = _deep_merge(default or {}, raw or {})
    return Expected(
        tool=data.get("tool"),
        intent=data.get("intent"),
        args=dict(data.get("args") or {}),
        response_contains=list(data.get("response_contains") or []),
        response_not_contains=list(data.get("response_not_contains") or []),
        state_assertions=dict(data.get("state_assertions") or {}),
    )


def _scoring_from(raw: dict[str, Any] | None, default: dict[str, Any] | None) -> ScoringConfig:
    data = _deep_merge(default or {}, raw or {})
    return ScoringConfig(
        weights=dict(data.get("weights") or {}),
        pass_threshold=float(data.get("pass_threshold", 0.8)),
        strict=bool(data.get("strict", False)),
    )


def _validate_expected(sample_id: str, expected: Expected) -> None:
    if not isinstance(expected.args, dict):
        raise DatasetValidationError(f"{sample_id}: expected.args must be a mapping")
    if not isinstance(expected.state_assertions, dict):
        raise DatasetValidationError(f"{sample_id}: expected.state_assertions must be a mapping")


def _sample_from(raw: dict[str, Any], defaults: dict[str, Any]) -> EvalSample:
    if not isinstance(raw, dict):
        raise DatasetValidationError("Each sample must be a mapping")
    sample_id = str(raw.get("id") or "").strip()
    if not sample_id:
        raise DatasetValidationError("Every sample must have an id")

    sample_type = str(raw.get("type") or "").strip()
    if sample_type not in {"single_turn", "multi_turn"}:
        raise DatasetValidationError(f"{sample_id}: type must be single_turn or multi_turn")

    tags = list(raw.get("tags", defaults.get("tags", [])) or [])
    initial_soft_state = _deep_merge(
        defaults.get("initial_soft_state") or {},
        raw.get("initial_soft_state") or {},
    )
    expected = _expected_from(raw.get("expected"), defaults.get("expected"))
    scoring = _scoring_from(raw.get("scoring"), defaults.get("scoring"))
    turns: list[EvalTurn] = []
    prompt = raw.get("prompt")

    if sample_type == "single_turn":
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetValidationError(f"{sample_id}: single_turn samples require prompt")
    else:
        raw_turns = raw.get("turns")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise DatasetValidationError(f"{sample_id}: multi_turn samples require turns")
        for idx, turn in enumerate(raw_turns, start=1):
            if not isinstance(turn, dict) or not str(turn.get("user") or "").strip():
                raise DatasetValidationError(f"{sample_id}: turn {idx} requires user")
            turn_expected = _expected_from(turn.get("expected"), defaults.get("expected"))
            _validate_expected(f"{sample_id} turn {idx}", turn_expected)
            turns.append(EvalTurn(user=str(turn["user"]), expected=turn_expected))

    _validate_expected(sample_id, expected)
    return EvalSample(
        id=sample_id,
        type=sample_type,
        tags=tags,
        prompt=prompt,
        expected=expected,
        turns=turns,
        scoring=scoring,
        initial_soft_state=initial_soft_state,
        allow_final_booking=bool(raw.get("allow_final_booking", False)),
    )


def validate_dataset(dataset: EvalDataset) -> None:
    if not dataset.samples:
        raise DatasetValidationError("Dataset must include at least one sample")
    seen: set[str] = set()
    for sample in dataset.samples:
        if sample.id in seen:
            raise DatasetValidationError(f"Duplicate sample id: {sample.id}")
        seen.add(sample.id)
        if not sample.tags:
            raise DatasetValidationError(f"{sample.id}: tags must not be empty")
        if sample.type == "multi_turn" and not sample.turns:
            raise DatasetValidationError(f"{sample.id}: multi_turn sample has no turns")


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> EvalDataset:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise DatasetValidationError("Dataset root must be a mapping")

    defaults = raw.get("defaults") or {}
    samples = [_sample_from(item, defaults) for item in raw.get("samples") or []]
    dataset = EvalDataset(
        version=str(raw.get("version") or ""),
        defaults=defaults,
        fixtures=dict(raw.get("fixtures") or {}),
        samples=samples,
        path=dataset_path,
    )
    validate_dataset(dataset)
    return dataset


def filter_samples(samples: list[EvalSample], tags: str | None) -> list[EvalSample]:
    if not tags:
        return samples
    wanted = {tag.strip() for tag in tags.split(",") if tag.strip()}
    return [sample for sample in samples if wanted.intersection(sample.tags)]
