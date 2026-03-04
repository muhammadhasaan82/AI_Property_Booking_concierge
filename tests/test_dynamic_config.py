import pytest
from pathlib import Path
import yaml
from services.dynamic_config import get_intent_catalog, get_vocabulary, get_routing_policies, get_guardrails, reload_all

def test_intent_catalog_load():
    cat = get_intent_catalog()
    assert cat is not None
    assert len(cat.intents) > 0
    assert "greeting" in cat.intents

def test_vocabulary_load():
    vocab = get_vocabulary()
    assert vocab is not None
    assert len(vocab.seed_property_types) > 0
    assert len(vocab.fallback_cities) > 0

def test_routing_policies_load():
    policies = get_routing_policies()
    assert policies is not None
    assert len(policies.sorted_policies) > 0

def test_guardrails_load():
    guard = get_guardrails()
    assert guard is not None
    assert len(guard.injection_patterns) > 0

def test_legacy_mode_fallback(monkeypatch):
    monkeypatch.setenv("LEGACY_RULES", "1")
    reload_all()
    # Vocabulary still works because LEGACY fallback is built-in
    vocab = get_vocabulary()
    res = vocab.fallback_cities
    assert len(res) > 0
    monkeypatch.setenv("LEGACY_RULES", "0")
    reload_all()
