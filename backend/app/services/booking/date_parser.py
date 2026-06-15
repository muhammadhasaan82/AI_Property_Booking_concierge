from __future__ import annotations
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.config.agent_config_loader import cfg
from app.config.booking_schema_loader import _reference_today
from app.services.booking.constants import (
    _CHECK_IN_DATE_PATTERN,
    _CHECK_OUT_DATE_PATTERN,
    _MONTHS,
)

def _infer_year_for_month_day(month: int, day: int) -> Optional[int]:
    """
    Infer a calendar year for a month/day pair using the schema reference date.

  When the month/day in the reference year is still today or later, that year is
  used; otherwise the next year is chosen so future bookings stay valid.
    """
    reference = _reference_today()
    try:
        this_year_date = date(reference.year, month, day)
    except ValueError:
        return None
    if this_year_date >= reference:
        return reference.year
    try:
        date(reference.year + 1, month, day)
        return reference.year + 1
    except ValueError:
        return reference.year

def _format_month_day(month: int, day: int, year: int) -> Optional[str]:
    try:
        return date(year, month, day).strftime(cfg.date_format)
    except ValueError:
        return None

def _parse_yearless_month_day(day: int, month_raw: str) -> Optional[str]:
    month = _MONTHS.get(str(month_raw).lower())
    if month is None:
        return None
    year = _infer_year_for_month_day(month, day)
    if year is None:
        return None
    return _format_month_day(month, day, year)

def _parse_single_structured_date(token: str) -> Optional[date]:
 
    m1 = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", token.strip())
    if m1:
        try:
            return date(int(m1.group(1)), int(m1.group(2)), int(m1.group(3)))
        except ValueError:
            return None

    m2 = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", token.strip())
    if m2:
        d = int(m2.group(1))
        m = int(m2.group(2))
        y = int(m2.group(3))
        try:
            return date(y, m, d)
        except ValueError:
            return None
    return None

def _parse_natural_date(text: str) -> Optional[str]:
    """
    Parse an ISO or common natural-language date from `text` and return it formatted using `cfg.date_format`.
    
    Searches for an ISO `YYYY-M-D` token or natural forms like "1st of January, 2025" and "January 1, 2025". If a valid date is found it is returned as a string formatted with `cfg.date_format`; otherwise `None` is returned.
    
    Parameters:
        text (str): Input text to scan for a date.
    
    Returns:
        Optional[str]: A date string formatted per `cfg.date_format` if parsing succeeds, `None` otherwise.
    """
    if not text:
        return None
    cleaned = re.sub(r"[\n\r]", " ", text).strip(" ,.;:")
    
    structured_match = re.search(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b", cleaned)
    if structured_match:
        parsed = _parse_single_structured_date(structured_match.group(0))
        if parsed:
            return parsed.strftime(cfg.date_format)
        else:
            return None

    patterns = [
        re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(?P<month>[a-z]+)\s*,?\s*(?P<year>\d{4})\b", re.I),
        re.compile(r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year>\d{4})\b", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        month_raw = str(match.group("month")).lower()
        month = _MONTHS.get(month_raw)
        if month is None:
            continue
        try:
            parsed = date(int(match.group("year")), month, int(match.group("day")))
            return parsed.strftime(cfg.date_format)
        except ValueError:
            return None

    yearless_patterns = (
        re.compile(
            r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(?P<month>[a-z]+)(?!\s*,?\s*\d{4})\b",
            re.I,
        ),
        re.compile(
            r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?!\s*,?\s*\d{4})\b",
            re.I,
        ),
    )
    for pattern in yearless_patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        parsed = _parse_yearless_month_day(int(match.group("day")), match.group("month"))
        if parsed:
            return parsed
    return None

def _find_all_dates(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract all dates mentioned in `text` and produce normalized date strings paired with their character spans.
    
    Parameters:
        text (str): Input text to scan for dates. An empty or falsy value yields an empty list.
    
    Returns:
        List[Tuple[str, int, int]]: A list of tuples `(date_str, start_index, end_index)` where `date_str` is formatted using `cfg.date_format`, and `start_index`/`end_index` are the character span of the matched date in the original text. The results are ordered by earliest start position and filtered so matches do not overlap. Recognizes ISO `YYYY-M-D` and common natural-language forms such as "1st of January, 2025" and "January 1, 2025".
    """
    if not text:
        return []
    results = []
    
    structured_pat = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b")
    for m in structured_pat.finditer(text):
        parsed = _parse_single_structured_date(m.group(0))
        if parsed:
            results.append((parsed.strftime(cfg.date_format), m.start(), m.end()))

    pat1 = re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(?P<month>[a-z]+)\s*,?\s*(?P<year>\d{4})\b", re.I)
    for m in pat1.finditer(text):
        month_raw = m.group("month").lower()
        month = _MONTHS.get(month_raw)
        if month is not None:
            try:
                parsed = date(int(m.group("year")), month, int(m.group("day")))
                results.append((parsed.strftime(cfg.date_format), m.start(), m.end()))
            except ValueError:
                pass

    pat2 = re.compile(r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<year>\d{4})\b", re.I)
    for m in pat2.finditer(text):
        month_raw = m.group("month").lower()
        month = _MONTHS.get(month_raw)
        if month is not None:
            try:
                parsed = date(int(m.group("year")), month, int(m.group("day")))
                results.append((parsed.strftime(cfg.date_format), m.start(), m.end()))
            except ValueError:
                pass

    pat3 = re.compile(
        r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(?P<month>[a-z]+)(?!\s*,?\s*\d{4})\b",
        re.I,
    )
    for m in pat3.finditer(text):
        month_raw = m.group("month").lower()
        month = _MONTHS.get(month_raw)
        if month is not None:
            year = _infer_year_for_month_day(month, int(m.group("day")))
            if year is not None:
                formatted = _format_month_day(month, int(m.group("day")), year)
                if formatted:
                    results.append((formatted, m.start(), m.end()))

    pat4 = re.compile(
        r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?!\s*,?\s*\d{4})\b",
        re.I,
    )
    for m in pat4.finditer(text):
        month_raw = m.group("month").lower()
        month = _MONTHS.get(month_raw)
        if month is not None:
            year = _infer_year_for_month_day(month, int(m.group("day")))
            if year is not None:
                formatted = _format_month_day(month, int(m.group("day")), year)
                if formatted:
                    results.append((formatted, m.start(), m.end()))

    results.sort(key=lambda x: (x[1], -(x[2] - x[1])))
    filtered = []
    last_end = -1
    for p_date, start, end in results:
        if start >= last_end:
            filtered.append((p_date, start, end))
            last_end = end
    return filtered

def _extract_dates_by_association(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Associate one or two date mentions in `text` with `check_in` and `check_out`.
    
    Attempts to identify date tokens in the input and assign them to check-in and check-out by proximity to explicit check-in/check-out phrases; when two or more dates are present it will prefer the closest matches and fall back to using the first two dates. If no dates are found, both return values are None.
    
    Parameters:
        text (str): Input text containing zero or more date mentions.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple `(check_in_date, check_out_date)` where each element is a normalized date string (as produced by the module's date-extraction helpers) or `None` when no appropriate date could be associated.
    """
    dates_found = _find_all_dates(text)
    if not dates_found:
        return None, None

    check_in_matches = list(re.finditer(_CHECK_IN_DATE_PATTERN, text, re.I))
    check_out_matches = list(re.finditer(_CHECK_OUT_DATE_PATTERN, text, re.I))

    if len(dates_found) >= 2 and check_in_matches and check_out_matches:
        first_date_start = dates_found[0][1]
        last_label_end = max(m.end() for m in check_in_matches + check_out_matches)
        if last_label_end <= first_date_start:
            return dates_found[0][0], dates_found[1][0]

    if len(dates_found) >= 2 and check_out_matches and not check_in_matches:
        last_check_out = max(check_out_matches, key=lambda match: match.start())
        check_out_date = None
        for d_val, d_start, _d_end in dates_found:
            if d_start >= last_check_out.end():
                check_out_date = d_val
                break
        if check_out_date:
            for d_val, _, _ in dates_found:
                if d_val != check_out_date:
                    return d_val, check_out_date

    check_in_date = None
    check_out_date = None
    
    if len(dates_found) == 1:
        d_val, d_start, d_end = dates_found[0]
        best_type = None
        min_dist = 999999
        for m in check_in_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 5
            if dist < min_dist:
                min_dist = dist
                best_type = "check_in"
        for m in check_out_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 10
            if dist < min_dist:
                min_dist = dist
                best_type = "check_out"
        if best_type == "check_in":
            check_in_date = d_val
        elif best_type == "check_out":
            check_out_date = d_val
        return check_in_date, check_out_date

    for d_val, d_start, d_end in dates_found:
        best_type = None
        min_dist = 999999
        for m in check_in_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 5
            if dist < min_dist:
                min_dist = dist
                best_type = "check_in"
        for m in check_out_matches:
            if m.end() <= d_start:
                dist = d_start - m.end()
            else:
                dist = (m.start() - d_end) * 10
            if dist < min_dist:
                min_dist = dist
                best_type = "check_out"
        
        if best_type == "check_in" and not check_in_date:
            check_in_date = d_val
        elif best_type == "check_out" and not check_out_date:
            check_out_date = d_val

    if len(dates_found) >= 2:
        if not check_in_date and not check_out_date:
            check_in_date = dates_found[0][0]
            check_out_date = dates_found[1][0]
        elif check_in_date and not check_out_date:
            for d_val, _, _ in dates_found:
                if d_val != check_in_date:
                    check_out_date = d_val
                    break
        elif check_out_date and not check_in_date:
            for d_val, _, _ in dates_found:
                if d_val != check_out_date:
                    check_in_date = d_val
                    break
                    
    return check_in_date, check_out_date

