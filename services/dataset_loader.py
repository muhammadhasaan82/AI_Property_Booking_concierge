# services/dataset_loader.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard-coded dataset loader for property listings → Chroma (via services.retrieval.bulk_upsert).

- Reads the CSV in 5,000-row chunks (low memory).
- Maps each row to your project's property schema expected by retrieval.py.
- Delegates embeddings + vector upserts to services.retrieval.bulk_upsert (Hugging Face + Chroma).
"""

from __future__ import annotations
import os
from typing import Dict, List

# IMPORTANT: keep this import; embeddings + Chroma config live in retrieval.py
try:
    from .retrieval import bulk_upsert
except ImportError:
    # Fallback for when running directly (not as a module)
    from retrieval import bulk_upsert

# If pandas isn't installed yet: uv add pandas
import pandas as pd

# ========= CHANGE THIS TO YOUR CSV LOCATION =========
# Example Windows absolute path:
DATASET_PATH = r"C:\Users\ASUS\Desktop\Calling-Agent-Chatbot\services\dataset.csv"
# ===================================================

# Optional: env used by services.retrieval (no changes needed here)
# CHROMA_DIR=./chroma
# EMBED_MODEL=thenlper/gte-small
# CHROMA_COLLECTION=properties


# ------------ helpers: row → property mapping ------------
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

# In dataset_loader.py, add debugging and fix the mapping:

def _map_row(row: Dict[str, str]) -> Dict:
    """
    Map CSV row → property dict your retrieval layer expects.
    """
    # Add some debugging
    property_id = row.get("id") or row.get("property_id") or row.get("uuid")
    if not property_id:
        print(f"WARNING: Skipping row with no ID: {row}")
        return None
        
    return {
        "id": str(property_id),
        "title": (row.get("title") or "Untitled Property").strip(),
        "city": (row.get("city") or row.get("location") or "").strip().lower(),  # Normalize to lowercase
        "country": (row.get("country") or "USA").strip(),
        "price_per_night": _to_int(row.get("price_per_night", 100)),  # Default to 100 if missing
        "bedrooms": _to_int(row.get("bedrooms", 1)),  # Default to 1 if missing
        "amenities": _parse_amenities(row.get("amenities", "")),
        "description": (row.get("description") or "").strip(),
    }

def run_ingestion(
    csv_path: str = DATASET_PATH,
    read_chunk_rows: int = 5000,
    upsert_batch_size: int = 500,  # Reduced batch size for stability
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
            # More robust validation
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
