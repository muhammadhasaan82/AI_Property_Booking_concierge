# Full Soft-Coding Refactor Plan for Hotel_Booking

You are a senior AI systems refactoring engineer. Your task is to convert this entire repository into a **maximally soft-coded, schema-driven, dynamic logic system** with minimal hardcoded assumptions.

## Objective
Refactor the project so the model/system can infer behavior from:
- runtime configuration,
- dataset/schema introspection,
- declarative rules,
- and LLM/tool metadata,

instead of fixed arrays, brittle keyword lists, and regex-heavy routing.

## What to fix (based on audit)

### 1) NLP and intent classification still rely heavily on hardcoded prototypes/phrases
- `services/nlp_engine.py` has large hardcoded intent prototypes and many lexical fallback sets/phrases.
- `services/nlp_engine.py` also includes many direct regex checks for intent/affirmation/status/property signals.

**Required refactor:**
- Move intent prototypes, thresholds, and fallback lexicons into external config (YAML/JSON) with hot-reload support.
- Add an embedding-based `IntentCatalog` that can be updated from config without code changes.
- Add confidence-calibrated routing with uncertainty handling (`ask_clarification`) instead of deterministic keyword forcing.
- Keep regex only for strict structural extraction (email/date/UUID), not intent semantics.

### 2) Central config is currently static and still hardcoded in source
- `services/config.py` contains fixed lists/sets for proceed/modify phrases, FAQ keywords, status/payment keywords, city fallbacks, property seeds, amenities synonyms.

**Required refactor:**
- Replace in-code constants with `config/*.yaml` driven settings.
- Build a typed config loader with validation (pydantic/dataclass + schema checks).
- Add environment override support and runtime reload endpoint.

### 3) Extractor partially dynamic but still has fixed fallback cities/aliases and seed property types
- `services/nlp_extractor.py` loads dataset dynamically but still merges hardcoded fallback vocab from config.

**Required refactor:**
- Create a vocabulary service that learns from:
  1. dataset columns,
  2. database values,
  3. optional admin-provided synonym tables.
- Remove code-level fallback city/property seeds (except optional bootstrap files).
- Add versioned vocabulary artifacts and cache invalidation.

### 4) Graph and agent routing contains fixed branch rules and keyword checks
- `services/graph.py` and `services/agents.py` use many explicit branch guards and hand-authored phrase checks.

**Required refactor:**
- Introduce a declarative policy/routing layer:
  - policies in YAML (state preconditions -> route),
  - dynamic priority scoring,
  - explainable route decisions.
- Keep graph nodes, but route selection must be policy-driven rather than handwritten condition chains.

### 5) Guardrails are regex-only static signatures
- `services/guardrails.py` uses fixed injection and leak regex patterns.

**Required refactor:**
- Keep deterministic regex as first pass, but add pluggable classifier/policy modules.
- Move guardrail patterns and actions to configuration with severity levels and audit logs.

### 6) Rust gateway intent inference uses fixed key arrays and FAQ keyword words
- `rust_gateway/src/gateway.rs` uses hardcoded keys/scores and hardcoded FAQ words.
- `rust_gateway/src/tools/search.rs` uses fixed key-based confidence and static trigger fields.

**Required refactor:**
- Move intent features and scoring weights to external TOML/YAML loaded at startup.
- Add schema-aware feature extraction:
  - infer tool intent from request schema + optional tool metadata,
  - avoid hardcoded key lists in Rust source.
- Add a capability registry where each tool self-describes required/optional fields and semantic hints.

## Deliverables
1. **Architecture update**
   - Add modules/services:
     - `dynamic_config_loader`
     - `intent_catalog`
     - `vocabulary_registry`
     - `routing_policy_engine`
     - `tool_capability_registry` (Python + Rust parity)

2. **Config files**
   - Create `config/intent_catalog.yaml`
   - Create `config/routing_policies.yaml`
   - Create `config/guardrails.yaml`
   - Create `rust_gateway/config/intent_features.toml`
   - Create `rust_gateway/config/tool_registry.toml`

3. **Backwards compatibility mode**
   - Add a transitional `LEGACY_RULES=true|false` flag.
   - Default to new dynamic mode with observability logs.

4. **Testing**
   - Add/expand tests for:
     - dynamic config reload,
     - intent classification after config update,
     - routing policy decisions,
     - Rust gateway intent inference from external config.
   - Include regression tests proving functionality without changing source constants.

5. **Observability**
   - Emit decision traces:
     - `intent_candidates`, `scores`, `selected_intent`, `policy_rule_id`, `confidence`, `fallback_reason`.
   - Add one debug endpoint to inspect active loaded schemas/configs.

## Non-negotiable acceptance criteria
- No business intent/routing logic should require Python/Rust source edits for normal behavior tuning.
- Adding a new intent/tool/policy should be possible via config + metadata only.
- Regex usage should be limited to structural parsing and security sanitization, not semantic intent heuristics.
- Hardcoded phrase lists in core runtime modules should be eliminated or replaced by externalized assets.

## Refactor approach (execute in phases)
1. Introduce config loaders and schemas.
2. Migrate NLP intent lists to `intent_catalog.yaml`.
3. Replace graph/agent branching with policy engine.
4. Externalize Rust gateway scoring to TOML.
5. Add compatibility adapter for legacy behavior.
6. Run full test suite and provide migration notes.

## Output format required from you
Return:
1. A concise change summary.
2. File-by-file diff rationale.
3. Risks/tradeoffs.
4. Remaining hardcoded hotspots (if any).
5. A migration checklist for production rollout.
