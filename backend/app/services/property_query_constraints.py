from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.config.property_search_filters_loader import (
    FilterConfig,
    get_filter_config,
    load_property_search_filters,
)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s+{}]+")
_NUMBER_RE = r"(?P<number>\d+(?:\.\d+)?)"


class SearchConstraint(BaseModel):
    field: str
    column: str
    operator: Literal["exact", "min", "max", "contains"]
    value: Any
    source_text: str


class PropertySearchQuery(BaseModel):
    city: Optional[str] = None
    property_type: Optional[str] = None
    constraints: List[SearchConstraint] = Field(default_factory=list)
    confidence: float = 0.0
    unresolved: List[str] = Field(default_factory=list)


def _normalize(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = lowered.replace("&", " and ")
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def _display_property_type(canonical: Optional[str]) -> Optional[str]:
    if not canonical:
        return None
    return str(canonical).strip().title()


def _runtime_dataset_rows() -> List[Dict[str, Any]]:
    try:
        from app.components import search as search_component

        rows = getattr(search_component, "_DATASET", None)
        return list(rows) if isinstance(rows, list) else []
    except Exception:
        return []


def _longest_match(normalized_message: str, candidates: Dict[str, str]) -> Optional[str]:
    best_value: Optional[str] = None
    best_len = -1
    for raw_term, resolved in candidates.items():
        term = _normalize(raw_term)
        if not term:
            continue
        if " " in term:
            matched = term in normalized_message
        else:
            matched = bool(re.search(rf"\b{re.escape(term)}\b", normalized_message))
        if matched and len(term) > best_len:
            best_len = len(term)
            best_value = resolved
    return best_value


def _city_candidates(city_filter: Optional[FilterConfig]) -> Dict[str, str]:
    candidates: Dict[str, str] = {}
    aliases = city_filter.aliases if city_filter and isinstance(city_filter.aliases, dict) else {}
    for alias, canonical in aliases.items():
        alias_text = str(alias).strip()
        canonical_text = str(canonical).strip()
        if alias_text and canonical_text:
            candidates[alias_text] = canonical_text
            candidates[canonical_text] = canonical_text

    for row in _runtime_dataset_rows():
        value = str(row.get("city") or "").strip()
        if value:
            candidates[value] = value
    return candidates


def _property_type_candidates(property_type_filter: Optional[FilterConfig]) -> Dict[str, str]:
    aliases = property_type_filter.taxonomy_aliases if property_type_filter else {}
    return {alias: canonical for alias, canonical in aliases.items() if alias and canonical}


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(_normalize(pattern))
    escaped = escaped.replace(re.escape("{number}"), _NUMBER_RE)
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def _numeric_value(raw: str, field_type: str) -> Any:
    value = float(raw)
    if field_type == "integer":
        return int(value)
    return int(value) if value.is_integer() else value


def _extract_numeric_constraints(
    normalized_message: str,
    field_name: str,
    filter_cfg: FilterConfig,
) -> List[SearchConstraint]:
    candidates: List[tuple[int, int, SearchConstraint]] = []
    for operator_name, operator_cfg in (filter_cfg.operators or {}).items():
        for pattern in operator_cfg.patterns:
            regex = _compile_pattern(pattern)
            for match in regex.finditer(normalized_message):
                value = _numeric_value(match.group("number"), filter_cfg.type)
                source_text = match.group(0).strip()
                candidate = SearchConstraint(
                    field=field_name,
                    column=filter_cfg.column,
                    operator=operator_name,
                    value=value,
                    source_text=source_text,
                )
                start, end = match.span()
                replaced = False
                for idx, (existing_start, existing_end, existing_constraint) in enumerate(candidates):
                    overlaps = start < existing_end and end > existing_start
                    if not overlaps:
                        continue
                    current_len = end - start
                    existing_len = existing_end - existing_start
                    if (
                        current_len > existing_len
                        or (
                            current_len == existing_len
                            and existing_constraint.operator == "exact"
                            and operator_name != "exact"
                        )
                    ):
                        candidates[idx] = (start, end, candidate)
                    replaced = True
                    break
                if not replaced:
                    candidates.append((start, end, candidate))
    return [constraint for _, _, constraint in candidates]


def _extract_amenity_constraints(
    normalized_message: str,
    field_name: str,
    filter_cfg: FilterConfig,
) -> List[SearchConstraint]:
    constraints: List[SearchConstraint] = []
    aliases = filter_cfg.aliases if isinstance(filter_cfg.aliases, dict) else {}
    seen_values: set[str] = set()
    for alias, canonical in sorted(
        aliases.items(),
        key=lambda item: len(_normalize(str(item[0]))),
        reverse=True,
    ):
        alias_norm = _normalize(str(alias))
        if not alias_norm:
            continue
        if " " in alias_norm:
            matched = alias_norm in normalized_message
        else:
            matched = bool(re.search(rf"\b{re.escape(alias_norm)}\b", normalized_message))
        if not matched:
            continue
        canonical_value = str(canonical).strip()
        if not canonical_value or canonical_value in seen_values:
            continue
        seen_values.add(canonical_value)
        constraints.append(
            SearchConstraint(
                field=field_name,
                column=filter_cfg.column,
                operator="contains",
                value=canonical_value,
                source_text=alias_norm,
            )
        )
    return constraints


def extract_property_search_query(message: str) -> PropertySearchQuery:
    normalized_message = _normalize(message)
    config = load_property_search_filters()

    city_filter = config.filters.get("city")
    property_type_filter = config.filters.get("property_type")

    city = _longest_match(normalized_message, _city_candidates(city_filter))
    property_type_canonical = _longest_match(
        normalized_message,
        _property_type_candidates(property_type_filter),
    )

    constraints: List[SearchConstraint] = []
    for field_name, filter_cfg in config.filters.items():
        if field_name in {"city", "property_type"}:
            continue
        if filter_cfg.type in {"integer", "number"}:
            constraints.extend(_extract_numeric_constraints(normalized_message, field_name, filter_cfg))
        elif filter_cfg.type == "list_contains":
            constraints.extend(_extract_amenity_constraints(normalized_message, field_name, filter_cfg))

    confidence = 0.0
    if city:
        confidence += 0.45
    if property_type_canonical:
        confidence += 0.25
    if constraints:
        confidence += min(0.10 * len(constraints), 0.30)

    return PropertySearchQuery(
        city=city,
        property_type=_display_property_type(property_type_canonical),
        constraints=constraints,
        confidence=min(confidence, 0.99),
        unresolved=[],
    )


def first_constraint(
    query: PropertySearchQuery,
    field: str,
    operator: Optional[Literal["exact", "min", "max", "contains"]] = None,
) -> Optional[SearchConstraint]:
    for constraint in query.constraints:
        if constraint.field != field:
            continue
        if operator is not None and constraint.operator != operator:
            continue
        return constraint
    return None


def field_column(field: str) -> Optional[str]:
    filter_cfg = get_filter_config(field)
    return filter_cfg.column if filter_cfg else None
