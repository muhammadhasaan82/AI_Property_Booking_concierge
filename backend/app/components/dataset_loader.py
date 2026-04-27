"""
Hard-coded dataset loader for property listings → Chroma (via services.retrieval.bulk_upsert).

- Reads the CSV in 5,000-row chunks (low memory).
- Maps each row to your project's property schema expected by retrieval.py.
- Delegates embeddings + vector upserts to services.retrieval.bulk_upsert (Hugging Face + Chroma).
"""

from __future__ import annotations
import os
from typing import Dict, List
try:
    from .retrieval import bulk_upsert
except ImportError:

    from retrieval import bulk_upsert

import pandas as pd
from pathlib import Path

try:
    from .config import DATASET_PATH as _CFG_DATASET_PATH
except ImportError:
    from config import DATASET_PATH as _CFG_DATASET_PATH


def _resolve_dataset_path() -> str:
    """Resolve dataset path from config, handling relative paths."""
    p = Path(_CFG_DATASET_PATH)
    if p.is_absolute() and p.exists():
        return str(p)

    root = Path(__file__).resolve().parents[3]
    candidate = root / _CFG_DATASET_PATH
    if candidate.exists():
        return str(candidate)

    local = Path(__file__).parent / "dataset.csv"
    if local.exists():
        return str(local)
    return _CFG_DATASET_PATH 


DATASET_PATH = _resolve_dataset_path()

def _to_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def _parse_amenities(val: str) -> List[str]:
    if not val:
        return []
    sep = ";" if ";" in val else ","
    return [a.strip().lower() for a in val.split(sep) if a.strip()]



def _map_row(row: Dict[str, str]) -> Dict:
    """
    Map CSV row → property dict your retrieval layer expects.
    """
 
    property_id = row.get("id") or row.get("property_id") or row.get("uuid")
    if not property_id:
        print(f"WARNING: Skipping row with no ID: {row}")
        return None
        
    return {
        "id": str(property_id),
        "title": (row.get("title") or "Untitled Property").strip(),
        "city": (row.get("city") or row.get("location") or "").strip().lower(),
        "country": (row.get("country") or "USA").strip(),
        "price_per_night": _to_int(row.get("price_per_night", 100)),
        "bedrooms": _to_int(row.get("bedrooms", 1)),
        "amenities": _parse_amenities(row.get("amenities", "")),
        "description": (row.get("description") or "").strip(),
    }

def run_ingestion(
    csv_path: str = DATASET_PATH,
    read_chunk_rows: int = 5000,
    upsert_batch_size: int = 500, 
) -> None:
    """
    Stream the CSV in chunks and load to vector store.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    total_rows = 0
    total_vectors = 0
    skipped = 0

    print(f"Starting ingestion from: {csv_path}")
    
    for df in pd.read_csv(csv_path, chunksize=read_chunk_rows, dtype=str, keep_default_na=False):
        mapped: List[Dict] = []
        for _, r in df.iterrows():
            prop = _map_row(r.to_dict())
            if prop is None:
                skipped += 1
                continue

            if not prop["id"] or not prop["title"]:
                skipped += 1
                continue
            mapped.append(prop)

        if mapped:
            print(f"Processing batch of {len(mapped)} properties...")
            for i in range(0, len(mapped), upsert_batch_size):
                batch = mapped[i : i + upsert_batch_size]
                bulk_upsert(batch)
                total_vectors += len(batch)
                print(f"  Upserted {len(batch)} properties (total: {total_vectors})")

        total_rows += len(df)
        print(f"Progress: {total_rows} rows read, {total_vectors} vectors indexed, {skipped} skipped")

    print(f"✅ Done! Total: {total_rows} rows, {total_vectors} indexed, {skipped} skipped")

if __name__ == "__main__":
    run_ingestion()
