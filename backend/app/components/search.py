from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..services.config import SEED_PROPERTY_TYPES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET_PATHS = [
    _REPO_ROOT / "backend" / "data" / "dataset.csv",      
    _REPO_ROOT / "data" / "dataset.csv",                 
    Path(__file__).parent / "dataset.csv",                
]

def _load_rows() -> List[Dict[str, Any]]:
    path: Optional[Path] = next((p for p in _DATASET_PATHS if p.exists()), None)
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            row["id"] = row.get("id") or row.get("property_id") or row.get("slug") or f"p_{len(rows)+1}"
            row["title"] = row.get("title") or row.get("name") or "Property"
            row["city"] = (row.get("city") or row.get("location") or "").strip()
            price_raw = row.get("price_per_night") or row.get("price") or row.get("nightly_price")
            try:
                row["price_per_night"] = int(float(str(price_raw).replace("$","").replace(",",""))) if price_raw else None
            except Exception:
                row["price_per_night"] = None
            row["property_type"] = row.get("property_type") or row.get("type") or ""
            for key in ("bedrooms", "bathrooms", "beds"):
                val = row.get(key)
                try:
                    row[key] = int(float(val)) if val not in (None, "", "null") else None
                except Exception:
                    row[key] = None
            am = row.get("amenities") or ""
            if isinstance(am, str):
                parts = [a.strip() for a in am.replace("|", ",").split(",") if a.strip()]
            else:
                parts = []
            row["amenities"] = parts
            try:
                row["rating"] = float(row.get("rating")) if row.get("rating") else None
            except Exception:
                row["rating"] = None
            row["description"] = row.get("description") or row.get("summary") or ""
            rows.append(row)
    return rows

_DATASET = _load_rows()

def _matches_location(row_city: str, wanted: Optional[str]) -> bool:
    if not wanted:
        return True
    a = (row_city or "").lower().strip()
    b = (wanted or "").lower().strip()

    if a == b:
        return True

    if len(b) > len(a):
        return b in a
    return False

def _amenity_subset(row_amenities: List[str], wanted: Optional[List[str]]) -> bool:
    if not wanted:
        return True
    rset = {a.strip().lower() for a in row_amenities or []}
    wset = {a.strip().lower() for a in wanted if a}
    return wset.issubset(rset)

def property_search(
    query_text: str,
    budget: int | None = None,
    amenities: List[str] | None = None,
    location: str | None = None,
    beds: int | None = None,
    property_type: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns a list of property dicts (see agents.py comments for required fields).
    """
    if not _DATASET:
        return []

    q = (query_text or "").lower()
    out: List[Dict[str, Any]] = []
    if not property_type:
        property_types = sorted(SEED_PROPERTY_TYPES)
        for pt in property_types:
            if pt in q:
                property_type = pt
                break
    
    for r in _DATASET:

        if not _matches_location(r.get("city",""), location):
            continue
        if property_type:
            row_type = (r.get("property_type") or "").lower()
            if property_type.lower() not in row_type and row_type not in property_type.lower():
                continue

        if budget is not None and r.get("price_per_night") is not None and r["price_per_night"] > budget:
            continue

        if beds is not None and r.get("beds") is not None and r["beds"] < beds:
            continue

        if not _amenity_subset(r.get("amenities", []), amenities):
            continue
        
       
        if q and not (location or property_type):
            hay = " ".join([str(r.get("title","")), str(r.get("property_type","")), str(r.get("city","")), str(r.get("description",""))]).lower()
            text_ok = any(tok in hay for tok in q.split())
            if not text_ok:
                continue

        out.append(r)

    out.sort(key=lambda x: (
        x.get("price_per_night") if x.get("price_per_night") is not None else 10**9,
        -(x.get("rating") or 0.0),
        x.get("title",""),
    ))
    
    return out


