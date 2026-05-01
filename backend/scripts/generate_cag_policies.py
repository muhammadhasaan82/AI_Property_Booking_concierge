"""
Generates rust_gateway/config/cag_policies.toml from data/faq_canonical.yaml.
 
This is the canonical pipeline for updating CAG knowledge:
  1. Edit data/faq_canonical.yaml.
  2. Run this script.
  3. POST /admin/reload-cag to the Rust gateway (no restart required).
 
Usage:
    python scripts/generate_cag_policies.py
    python scripts/generate_cag_policies.py --dry-run     
    python scripts/generate_cag_policies.py --paraphrase    
    python scripts/generate_cag_policies.py --backup   
"""
from __future__ import annotations
import argparse
import difflib
import keyword
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("generate_cag_policies")

ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = ROOT / "data" / "faq_canonical.yaml"
TARGET_TOML = ROOT / "rust_gateway" / "config" / "cag_policies.toml"

def _toml_escape_string(value: str) -> str:
    """Esacpe a string for TOML basic string syntax."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()

def _render_string_array(items: List[str]) -> str:
    rendered = ", ".join(f'"{_toml_escape_string(s)}"' for s in items)
    return f"[{rendered}]"

def _render_canonical_questions_block(items: List[str]) -> str:
    if not items:
        return "[]"
    lines = ["["]
    for s in items:
        lines.append(f'     "{_toml_escape_string(s)}",')
    lines.append("]")
    return "\n".join(lines)

def render_toml(data: Dict[str ,Any]) -> str:
    settings = data.get("settings") or {}
    policies = data.get("policies") or []
    
    lines: List[str] = [
        "# AUTO-GENERATED — DO NOT EDIT BY HAND.",
        "# Source of truth: backend/data/faq_canonical.yaml",
        "# Regenerate via: python scripts/generate_cag_policies.py",
        "",
        "[settings]",
        f'keyword_threshold = {float(settings.get("keyword_threshold", 0.6))}',
        f'fuzzy_threshold = {float(settings.get("fuzzy_threshold", 0.82))}',
        f'ttl_seconds = {int(settings.get("ttl_seconds", 3600))}',
        "",
    ]

    for pol in policies:
        pol_id = str(pol.get("id", "")).strip()
        if not pol_id:
            logger.warning("Skipping policy with no id: %s", pol)
            continue
        answer = str(pol.get("answer", "")).strip()
        canonical = [pol.get("canonical_question", "")] + list(pol.get("parameters") or [])
        canonical = [str(s).strip() for s in canonical if s and str(s).strip()]
        keywords = [str(k).strip() for k in (pol.get("keywords") or []) if k]

        lines.append("[[policies]]")
        lines.append(f'id = "{_toml_escape_string(pol_id)}"')
        lines.append(f'answer = "{_toml_escape_string(answer)}"')
        lines.append(f"keywords = {_render_string_array(keywords)}")
        lines.append("canonical_questions = "+ _render_canonical_questions_block(canonical))
        lines.append("")
        
    return "\n".join(lines).rstrip() + "\n"

def expand_paraphrases(data: Dict[str, Any], max_per_policy: int = 4) -> Dict[str,Any]:
    """Use the configured dispatcher LLM to add paraphrases variants to each policy.

    Idempotent — never removes existing paraphrases. New ones are deduplicated.
    """
    try:
        import litellm
        from app.config.agent_config_loader import cfg
    except Exception as exc:
        logger.warning("LLM expansion skipped (litellm/cfg unavailable): %s", exc)
        return data

    model = cfg.dispatcher_model
    for pol in data.get("policies") or []:
        canonical = pol.get("canonical_question", "")
        if not canonical:
            continue
        existing = set(p.lower().strip() for p in (pol.get("paraphrases") or []))

        prompt = (
            "Generate up to {n} short, natural paraphrases of this question. "
            "Return them as a plain bullet list (one per line, no numbering, no quotes). "
            "Do not change the meaning. Only output the list — no preamble.\n\n"
            "Question: {q}"
        ).format(n=max_per_policy, q=canonical)
        
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200,
            )
            text = resp["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            logger.warning("LLM paraphrase failed for %s: %s", pol.get("id"), exc)
            continue

        new_paraphrases: List[str] = []
        for line in text.splitlines():
            cleaned = line.strip().lstrip("-•*").strip('"').strip()
            if not cleaned:
                continue
            if cleaned.lower() == canonical.lower():
                continue
            if cleaned.lower() in existing:
                continue
            existing.add(cleaned.lower())
            new_paraphrases.append(cleaned)
        
        if new_paraphrases:
            pol["paraphrases"] = list(pol.get("paraphrases") or []) + new_paraphrases
            logger.info("[%s] +%d paraphrases", pol.get("id"), len(new_paraphrases))

    return data

def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate cag_policies from faq_canonical.yaml")
    parser.add_argument("--dry-run", action="store_true", help="show diff but do not write")
    parser.add_argument("--paraphrase", action="store_true", help="expand paraphrase via LLM")
    parser.add_argument("--backup", action="store_true", help="save existing TOML as .bak")
    parser.add_argument("--source", type=Path, default=SOURCE_YAML)
    parser.add_argument("--target", type=Path, default=TARGET_TOML)
    args = parser.parse_args()
    
    if not args.source.exists():
        logger.error("Source file not found: %s", args.source)
        return 1

    with open(args.source, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if args.paraphrase:
        sys.path.insert(0, str(ROOT))
        data = expand_paraphrases(data)
    
    new_toml = render_toml(data)

    old_toml = ""
    if args.target.exists():
        with open(args.target, "r", encoding="utf-8") as f:
            old_toml = f.read()

    if old_toml.strip() == new_toml.strip():
        logger.info("No changes - %s already up to date.", args.target)
        return 0
    
    diff = list(difflib.unified_diff(
        old_toml.splitlines(keepends=True),
        new_toml.splitlines(keepends=True),
        fromfile=str(args.target) + " (current)",
        tofile=str(args.target) + " (new)",
        n=3
    ))
    if diff:
        sys.stdout.writelines(diff)

    if args.dry_run:
        logger.info("[dry-run] not writing.")
        return 0

    if args.backup and args.target.exists():
        backup_path = args.target.with_suffix(args.target.suffix + ".bak")
        shutil.copy2(args.target, backup_path)
        logger.info("Backed Saved: %s", backup_path)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    with open(args.target, "w", encoding="utf-8") as f:
        f.write(new_toml)
    logger.info("Wrote %s (%d policies)", args.target, len(data.get("policies") or []))
    return 0

if __name__ == "__main__":
    sys.exit(main())