"""
YAML-driven service coverage guard.

Region detection terms (cities, country aliases, codes) are loaded from
service_coverage.yaml. Python only normalizes text, matches configured terms,
and applies deterministic supported/unsupported routing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "service_coverage.yaml"


@dataclass(frozen=True)
class CoverageDecision:
    supported: bool
    blocked: bool
    requested_region: str | None
    country: str | None
    message: str | None
    supported_countries: list[str]


@dataclass(frozen=True)
class ServiceCoverageConfig:
    version: str
    enabled: bool
    supported_countries: tuple[str, ...]
    supported_country_codes: tuple[str, ...]
    unsupported_region_response: str
    allow_dataset_regions_outside_supported_market: bool


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _contains_term(normalized_message: str, term: str) -> bool:
    needle = _normalize(term)
    if not needle:
        return False
    if " " in needle:
        return needle in normalized_message
    return bool(re.search(rf"\b{re.escape(needle)}\b", normalized_message))


class _ServiceCoverageRouter:
    def __init__(self, raw: Dict[str, Any]) -> None:
        raw = raw if isinstance(raw, dict) else {}
        self.version = str(raw.get("version", "1.0"))

        coverage = raw.get("coverage") or {}
        if not isinstance(coverage, dict):
            coverage = {}

        self.config = ServiceCoverageConfig(
            version=self.version,
            enabled=bool(coverage.get("enabled", True)),
            supported_countries=tuple(_as_str_list(coverage.get("supported_countries"))),
            supported_country_codes=tuple(
                code.upper() for code in _as_str_list(coverage.get("supported_country_codes"))
            ),
            unsupported_region_response=str(
                coverage.get("unsupported_region_response") or ""
            ).strip(),
            allow_dataset_regions_outside_supported_market=bool(
                coverage.get("allow_dataset_regions_outside_supported_market", False)
            ),
        )

        city_map = raw.get("city_country_map") or {}
        self._city_terms: List[tuple[str, str]] = []
        if isinstance(city_map, dict):
            for city, country in city_map.items():
                city_name = str(city).strip()
                country_name = str(country).strip()
                if city_name and country_name:
                    self._city_terms.append((city_name, country_name))
        self._city_terms.sort(key=lambda item: len(item[0]), reverse=True)

        aliases = raw.get("region_aliases") or {}
        self._alias_to_country: Dict[str, str] = {}
        self._country_canonical: Dict[str, str] = {}
        if isinstance(aliases, dict):
            for country, alias_list in aliases.items():
                country_name = str(country).strip()
                if not country_name:
                    continue
                canonical = country_name
                self._country_canonical[_normalize(country_name)] = canonical
                for alias in _as_str_list(alias_list):
                    self._alias_to_country[_normalize(alias)] = canonical
                self._alias_to_country[_normalize(country_name)] = canonical

        for country in self.config.supported_countries:
            self._country_canonical[_normalize(country)] = country

        self._supported_countries_norm = {
            _normalize(country) for country in self.config.supported_countries
        }
        self._supported_codes = set(self.config.supported_country_codes)

    def detect_region_in_message(self, message: str) -> Optional[str]:
        """
        Return the canonical country name when a configured city or region alias
        appears in the message; otherwise None.
        """
        normalized = _normalize(message)
        if not normalized:
            return None

        for city, country in self._city_terms:
            if _contains_term(normalized, city):
                return self._canonical_country(country)

        alias_terms = sorted(self._alias_to_country.keys(), key=len, reverse=True)
        for alias in alias_terms:
            if _contains_term(normalized, alias):
                return self._alias_to_country[alias]

        return None

    def check_region_supported(self, region: str) -> CoverageDecision:
        """
        Evaluate whether a detected canonical country is within configured coverage.
        """
        supported_countries = list(self.config.supported_countries)
        country = self._canonical_country(region)
        requested = (region or "").strip() or None

        if not self.config.enabled:
            return CoverageDecision(
                supported=True,
                blocked=False,
                requested_region=requested,
                country=country,
                message=None,
                supported_countries=supported_countries,
            )

        if not country:
            return CoverageDecision(
                supported=True,
                blocked=False,
                requested_region=requested,
                country=None,
                message=None,
                supported_countries=supported_countries,
            )

        if self._is_supported_country(country):
            return CoverageDecision(
                supported=True,
                blocked=False,
                requested_region=requested,
                country=country,
                message=None,
                supported_countries=supported_countries,
            )

        blocked_message = self.config.unsupported_region_response or None
        return CoverageDecision(
            supported=False,
            blocked=True,
            requested_region=requested,
            country=country,
            message=blocked_message,
            supported_countries=supported_countries,
        )

    def evaluate_message(self, message: str) -> CoverageDecision:
        detected = self.detect_region_in_message(message)
        if not detected:
            return CoverageDecision(
                supported=True,
                blocked=False,
                requested_region=None,
                country=None,
                message=None,
                supported_countries=list(self.config.supported_countries),
            )
        return self.check_region_supported(detected)

    def _canonical_country(self, region: str) -> Optional[str]:
        key = _normalize(region)
        if not key:
            return None
        if key in self._country_canonical:
            return self._country_canonical[key]
        upper = region.strip().upper()
        if upper in self._supported_codes:
            for country in self.config.supported_countries:
                if _normalize(country) in self._country_canonical:
                    return self._country_canonical[_normalize(country)]
        return region.strip() or None

    def _is_supported_country(self, country: str) -> bool:
        normalized = _normalize(country)
        if normalized in self._supported_countries_norm:
            return True
        upper = country.strip().upper()
        return upper in self._supported_codes


def _load_router(raw: Optional[Dict[str, Any]] = None) -> _ServiceCoverageRouter:
    if raw is not None:
        return _ServiceCoverageRouter(raw)
    if not _CONFIG_PATH.exists():
        logger.warning("[service_coverage] %s missing", _CONFIG_PATH)
        return _ServiceCoverageRouter({})
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return _ServiceCoverageRouter(yaml.safe_load(handle) or {})


_router: _ServiceCoverageRouter = _load_router()


def reload_service_coverage() -> None:
    """Reload service coverage configuration from disk."""
    global _router
    _router = _load_router()


def load_service_coverage(raw: Optional[Dict[str, Any]] = None) -> ServiceCoverageConfig:
    """
    Return the active service coverage configuration.

    When `raw` is provided, build a temporary router (used in tests) without
    mutating the module-level singleton.
    """
    router = _load_router(raw) if raw is not None else _router
    return router.config


def get_service_coverage_snapshot() -> dict[str, Any]:
    """
    Secret-safe debug snapshot for /debug/config.
    """
    cfg = _router.config
    return {
        "enabled": cfg.enabled,
        "supported_countries": list(cfg.supported_countries),
        "allow_dataset_regions_outside_supported_market": (
            cfg.allow_dataset_regions_outside_supported_market
        ),
    }


def detect_region_in_message(message: str) -> Optional[str]:
    return _router.detect_region_in_message(message)


def check_region_supported(region: str) -> CoverageDecision:
    return _router.check_region_supported(region)


def evaluate_message_coverage(message: str) -> CoverageDecision:
    return _router.evaluate_message(message)


def _set_router_for_tests(router: _ServiceCoverageRouter) -> None:
    """Test-only hook to swap the module router."""
    global _router
    _router = router
