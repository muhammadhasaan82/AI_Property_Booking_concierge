from __future__ import annotations
"""
Schema-driven search planner.

Orchestrates the full search flow:
1. Extract constraints from user message using schema
2. Merge with previous session constraints
3. Enforce hard filters (exact matches) before ranking
4. Apply soft filters (fuzzy matches) for ranking
5. Provide detailed trace logging for debugging

This replaces hardcoded search logic with a dynamic, schema-driven approach.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.dynamic_constraints import DynamicConstraints, create_constraints_from_dict
from app.services.property_schema import PropertySchema, get_property_schema
from app.config.property_search_filters_loader import FilterConfig, load_property_search_filters

logger = logging.getLogger(__name__)


def _sort_key(value: Any) -> Tuple[int, Any]:
    if value is None or value == "":
        return (1, 0)
    if isinstance(value, (int, float)):
        return (0, value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return (0, float(stripped.replace("$", "").replace(",", "")))
        except ValueError:
            return (0, stripped.lower())
    return (0, str(value))


@dataclass
class SearchTrace:
    """Trace log for search execution."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = "unknown"
    user_message: str = ""
    message: str = ""
    extracted_constraints: Dict[str, Any] = field(default_factory=dict)
    merged_constraints: Dict[str, Any] = field(default_factory=dict)
    hard_filters: Dict[str, Any] = field(default_factory=dict)
    soft_filters: Dict[str, Any] = field(default_factory=dict)
    sort_preferences: List[Dict[str, str]] = field(default_factory=list)
    pre_filter_count: int = 0
    post_filter_count: int = 0
    final_count: int = 0
    filter_effectiveness: float = 0.0
    search_path: str = ""
    
    def log(self, msg: str) -> None:
        """Add a log message."""
        self.message += f"[{datetime.now().isoformat()}] {msg}\n"
        logger.info(f"[SearchTrace] {msg}")
    
    def calculate_effectiveness(self) -> None:
        """Calculate filter effectiveness percentage."""
        if self.pre_filter_count > 0:
            self.filter_effectiveness = (
                (self.pre_filter_count - self.post_filter_count) / self.pre_filter_count
            ) * 100


@dataclass
class SearchPlan:
    """
    Schema-driven search plan.
    
    Contains extracted constraints, merged with session state,
    ready for enforcement and ranking.
    """
    constraints: DynamicConstraints
    trace: SearchTrace
    session_id: Optional[str] = None
    user_message: str = ""
    sort_preferences: List[Dict[str, str]] = field(default_factory=list)
    
    def get_hard_filters(self) -> Dict[str, Any]:
        """Get hard filters for enforcement."""
        return self.constraints.get_hard_filters()
    
    def get_soft_filters(self) -> Dict[str, Any]:
        """Get soft filters for ranking."""
        return self.constraints.get_soft_filters()


class SearchPlanner:
    """
    Schema-driven search planner.
    
    Replaces hardcoded search logic with dynamic constraint extraction,
    merging, and enforcement based on dataset schema.
    """
    
    def __init__(self, schema: Optional[PropertySchema] = None):
        """
        Initialize search planner.
        
        Args:
            schema: Property schema for validation. If None, uses global schema.
        """
        self._schema = schema or get_property_schema()
        self._extractors: Dict[str, Callable] = {}
        self._sort_phrase_map: Dict[str, Dict[str, List[str]]] = {}
        self._register_schema_extractors()
    
    def _register_schema_extractors(self) -> None:
        """Register extractors dynamically from configured searchable filters."""
        filters_cfg = load_property_search_filters()
        for field_name, filter_cfg in filters_cfg.filters.items():
            if not self._schema.is_searchable(field_name):
                logger.debug("Skipping extractor for non-searchable schema field: %s", field_name)
                continue
            extractor = self._build_extractor(field_name, filter_cfg)
            if extractor is not None:
                self._extractors[field_name] = extractor
            if filter_cfg.sort_phrases:
                self._sort_phrase_map[field_name] = {
                    direction: [phrase.strip().lower() for phrase in phrases if phrase.strip()]
                    for direction, phrases in filter_cfg.sort_phrases.items()
                }

    def _build_extractor(
        self,
        field_name: str,
        filter_cfg: FilterConfig,
    ) -> Optional[Callable[[str], Optional[Tuple[Any, str]]]]:
        if field_name == "city":
            return self._build_city_extractor()
        if filter_cfg.type == "taxonomy":
            return self._build_taxonomy_extractor(filter_cfg)
        if filter_cfg.type in {"integer", "number"}:
            return self._build_numeric_extractor(filter_cfg)
        if filter_cfg.type == "list_contains":
            return self._build_list_contains_extractor(filter_cfg)
        return None

    def _build_city_extractor(self) -> Callable[[str], Optional[Tuple[str, str]]]:
        def extract_city(message: str) -> Optional[Tuple[str, str]]:
            from app.services.direct_property_search import resolve_supported_city_from_message

            city_match = resolve_supported_city_from_message(message)
            if city_match.city:
                return city_match.city, "exact"
            return None

        return extract_city

    def _build_taxonomy_extractor(
        self,
        filter_cfg: FilterConfig,
    ) -> Callable[[str], Optional[Tuple[str, str]]]:
        aliases = dict(filter_cfg.taxonomy_aliases or {})

        def extract_taxonomy(message: str) -> Optional[Tuple[str, str]]:
            normalized_message = str(message or "").strip().lower()
            if not normalized_message:
                return None

            from app.services.direct_property_search import extract_property_type_from_message

            direct_value = extract_property_type_from_message(message)
            if direct_value:
                return str(direct_value).strip().lower(), "exact"

            best_match = None
            best_len = 0
            for alias, canonical in aliases.items():
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, normalized_message):
                    if len(alias) > best_len:
                        best_match = str(canonical).strip().lower()
                        best_len = len(alias)
            if best_match:
                return best_match, "exact"
            return None

        return extract_taxonomy

    def _build_numeric_extractor(
        self,
        filter_cfg: FilterConfig,
    ) -> Callable[[str], Optional[Tuple[Any, str]]]:
        compiled: List[Tuple[str, re.Pattern[str]]] = []
        for operator_name, operator_cfg in filter_cfg.operators.items():
            for template in operator_cfg.patterns:
                pattern = re.escape(template).replace(
    r"\{number\}",
    r"(?:[^\w\s]\s*)?(\d+(?:\.\d+)?)",
)
                pattern = pattern.replace(r"\ ", r"\s+")
                compiled.append((operator_name, re.compile(rf"\b{pattern}\b", re.IGNORECASE)))

        def extract_numeric(message: str) -> Optional[Tuple[Any, str]]:
            normalized_message = str(message or "").strip()
            if not normalized_message:
                return None
            best_match: Optional[Tuple[str, re.Match[str]]] = None
            best_score: Optional[Tuple[int, int]] = None
            for operator_name, pattern in compiled:
                match = pattern.search(normalized_message)
                if not match:
                    continue
                span = match.span(0)
                score = (span[1] - span[0], -span[0])
                if best_score is None or score > best_score:
                    best_match = (operator_name, match)
                    best_score = score
            if best_match is None:
                return None
            operator_name, match = best_match
            raw_value = match.group(1)
            if filter_cfg.type == "integer":
                return int(float(raw_value)), operator_name
            return float(raw_value), operator_name

        return extract_numeric

    def _build_list_contains_extractor(
        self,
        filter_cfg: FilterConfig,
    ) -> Callable[[str], Optional[Tuple[List[str], str]]]:
        aliases = {
            str(alias).strip().lower(): str(value).strip().lower()
            for alias, value in (filter_cfg.aliases or {}).items()
            if str(alias).strip() and str(value).strip()
        }

        def extract_list_contains(message: str) -> Optional[Tuple[List[str], str]]:
            normalized_message = str(message or "").strip().lower()
            if not normalized_message:
                return None
            matches: List[str] = []
            for phrase, canonical in aliases.items():
                if phrase in normalized_message and canonical not in matches:
                    matches.append(canonical)
            if matches:
                return matches, "contains"
            return None

        return extract_list_contains

    def extract_sort_preferences(self, message: str) -> List[Dict[str, str]]:
        normalized_message = str(message or "").strip().lower()
        if not normalized_message:
            return []

        preferences: List[Dict[str, str]] = []
        for field_name, directions in self._sort_phrase_map.items():
            for direction, phrases in directions.items():
                if any(phrase in normalized_message for phrase in phrases):
                    preferences.append({"field": field_name, "direction": direction})
                    break
        return preferences
    
    def extract_constraints(self, message: str) -> DynamicConstraints:
        """
        Extract constraints from user message using registered extractors.
        
        Args:
            message: User's natural language message
        
        Returns:
            DynamicConstraints with extracted values
        """
        constraints = DynamicConstraints(schema=self._schema)
        
        for field_name, extractor in self._extractors.items():
            try:
                result = extractor(message)
                if result:
                    value, operator = result
                    constraints.set(field_name, value, operator)
            except Exception as exc:
                logger.warning("Failed to extract %s: %s", field_name, exc)
        
        return constraints
    
    def plan_search(
        self,
        message: str,
        session_constraints: Optional[DynamicConstraints] = None,
        session_id: Optional[str] = None
    ) -> SearchPlan:
        """
        Create a search plan from user message and session state.
        
        Args:
            message: User's natural language message
            session_constraints: Previous constraints from session
            session_id: Session identifier for logging
        
        Returns:
            SearchPlan ready for execution
        """
        trace = SearchTrace(session_id=session_id or "unknown", user_message=message)
        trace.log(f"Planning search for message: '{message}'")
        
        extracted = self.extract_constraints(message)
        sort_preferences = self.extract_sort_preferences(message)
        trace.extracted_constraints = extracted.to_dict()
        trace.log(f"Extracted constraints: {extracted.to_dict()}")
        if sort_preferences:
            trace.sort_preferences = list(sort_preferences)
            trace.log(f"Sort preferences: {sort_preferences}")
        
        if session_constraints and not session_constraints.is_empty():
            merged = session_constraints.merge_with(extracted, override=True)
            trace.log(f"Merged with session constraints: {merged.to_dict()}")
        else:
            merged = extracted
            trace.log("No session constraints to merge")
        
        trace.merged_constraints = merged.to_dict()
        
        hard_filters = merged.get_hard_filters()
        soft_filters = merged.get_soft_filters()
        
        trace.hard_filters = hard_filters
        trace.soft_filters = soft_filters
        trace.log(f"Hard filters: {hard_filters}")
        trace.log(f"Soft filters: {soft_filters}")
        
        return SearchPlan(
            constraints=merged,
            trace=trace,
            session_id=session_id,
            user_message=message,
            sort_preferences=sort_preferences,
        )
    
    def enforce_hard_filters(
        self,
        properties: List[Dict[str, Any]],
        plan: SearchPlan
    ) -> List[Dict[str, Any]]:
        """
        Enforce hard filters on property list.
        
        Hard filters are exact matches that must be satisfied.
        Properties that don't match are excluded.
        
        Args:
            properties: List of property dictionaries
            plan: Search plan with constraints
        
        Returns:
            Filtered list of properties
        """
        hard_filters = plan.get_hard_filters()
        
        if not hard_filters:
            plan.trace.log("No hard filters to enforce")
            plan.trace.pre_filter_count = len(properties)
            plan.trace.post_filter_count = len(properties)
            plan.trace.final_count = len(properties)
            plan.trace.calculate_effectiveness()
            return properties
        
        plan.trace.pre_filter_count = len(properties)
        plan.trace.log(f"Enforcing hard filters on {len(properties)} properties")
        
        filtered = []
        for prop in properties:
            if self._matches_hard_filters(prop, hard_filters):
                filtered.append(prop)
        
        plan.trace.post_filter_count = len(filtered)
        plan.trace.log(f"After hard filters: {len(filtered)} properties remain")
        plan.trace.calculate_effectiveness()
        
        return filtered
    
    def _matches_hard_filters(
        self,
        prop: Dict[str, Any],
        hard_filters: Dict[str, Any]
    ) -> bool:
        """
        Check if a property matches all hard filters.
        
        Args:
            prop: Property dictionary
            hard_filters: Hard filter constraints
        
        Returns:
            True if property matches all hard filters
        """
        for field_name, filter_spec in hard_filters.items():
            value = filter_spec["value"]
            operator = filter_spec["operator"]
            
            prop_value = prop.get(field_name)
            
            if operator == "exact":
                if isinstance(value, str) and isinstance(prop_value, str):
                    if value.lower() != prop_value.lower():
                        return False
                elif value != prop_value:
                    return False
            
            elif operator == "contains":
                if isinstance(prop_value, list):
                    if isinstance(value, list):
                        normalized_prop = {str(v).strip().lower() for v in prop_value}
                        if not all(str(v).strip().lower() in normalized_prop for v in value):
                            return False
                    else:
                        normalized_prop = {str(v).strip().lower() for v in prop_value}
                        if str(value).strip().lower() not in normalized_prop:
                            return False
                else:
                    return False
            
            elif operator == "min":
                try:
                    if float(prop_value or 0) < float(value):
                        return False
                except (ValueError, TypeError):
                    return False
            
            elif operator == "max":
                try:
                    if float(prop_value or float('inf')) > float(value):
                        return False
                except (ValueError, TypeError):
                    return False
        
        return True
    
    def apply_soft_filters(
        self,
        properties: List[Dict[str, Any]],
        plan: SearchPlan
    ) -> List[Dict[str, Any]]:
        """
        Apply soft filters for ranking (not filtering).
        
        Soft filters influence ranking but don't exclude results.
        
        Args:
            properties: List of property dictionaries
            plan: Search plan with constraints
        
        Returns:
            Ranked list of properties
        """
        soft_filters = plan.get_soft_filters()
        
        if not soft_filters and not plan.sort_preferences:
            plan.trace.log("No soft filters to apply")
            plan.trace.final_count = len(properties)
            return properties
        
        plan.trace.log("Applying soft filters for ranking")
        
        def relevance_score(prop: Dict[str, Any]) -> float:
            score = 0.0
            for field_name, filter_spec in soft_filters.items():
                value = filter_spec["value"]
                prop_value = prop.get(field_name, "")
                
                if isinstance(value, str) and isinstance(prop_value, str):
                    if value.lower() in prop_value.lower():
                        score += 1.0
            
            return score

        ranked = sorted(properties, key=relevance_score, reverse=True)
        for preference in reversed(plan.sort_preferences):
            field_name = str(preference.get("field") or "").strip()
            direction = str(preference.get("direction") or "asc").strip().lower()
            if not field_name:
                continue
            ranked = sorted(
                ranked,
                key=lambda item, sort_field=field_name: _sort_key(item.get(sort_field)),
                reverse=direction == "desc",
            )
        plan.trace.final_count = len(ranked)
        
        return ranked
    
    def execute_search(
        self,
        properties: List[Dict[str, Any]],
        plan: SearchPlan
    ) -> Tuple[List[Dict[str, Any]], SearchTrace]:
        """
        Execute search plan on property list.
        
        Args:
            properties: List of property dictionaries
            plan: Search plan with constraints
        
        Returns:
            Tuple of (filtered/ranked properties, trace log)
        """
        plan.trace.log(f"Executing search on {len(properties)} properties")
        
        filtered = self.enforce_hard_filters(properties, plan)
        
        ranked = self.apply_soft_filters(filtered, plan)
        
        plan.trace.log(f"Search complete: {len(ranked)} properties returned")
        
        return ranked, plan.trace


_planner_instance: Optional[SearchPlanner] = None


def get_search_planner(schema: Optional[PropertySchema] = None) -> SearchPlanner:
    """Get or create the global search planner instance."""
    global _planner_instance
    
    if _planner_instance is None:
        _planner_instance = SearchPlanner(schema)
    
    return _planner_instance
