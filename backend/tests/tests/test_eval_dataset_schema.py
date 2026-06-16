from __future__ import annotations

from pathlib import Path

import yaml

from evaluation.eval_dataset import load_dataset


DATASET = Path("evaluation/golden_set.yaml")


def test_golden_set_yaml_parses_with_safe_load():
    data = yaml.safe_load(DATASET.read_text(encoding="utf-8"))
    assert data["version"] == "2.0"
    assert isinstance(data["samples"], list)


def test_dataset_loader_validates_unique_ids_and_required_fields():
    dataset = load_dataset(DATASET)
    ids = [sample.id for sample in dataset.samples]
    assert len(ids) == len(set(ids))
    assert dataset.samples
    for sample in dataset.samples:
        assert sample.id
        assert sample.type in {"single_turn", "multi_turn"}
        assert sample.tags
        assert sample.expected is not None


def test_multi_turn_samples_validate_turns():
    dataset = load_dataset(DATASET)
    multi_turn = [sample for sample in dataset.samples if sample.type == "multi_turn"]
    assert multi_turn
    for sample in multi_turn:
        assert sample.turns
        for turn in sample.turns:
            assert turn.user
            assert turn.expected is not None
