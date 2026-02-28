# services/retrieval.py
from __future__ import annotations
import os
import json
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ---------- Config ----------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "thenlper/gte-small")  # e.g., 'BAAI/bge-small-en-v1.5'
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "properties")

# ---------- Lazy imports / feature flags ----------
_HAS_VECTOR = True
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except Exception as _e:
    print(f"[retrieval] Vector mode unavailable ({_e}); using JSON fallback.")
    _HAS_VECTOR = False

# ---------- Fallback storage (JSON) ----------
_properties_file = Path(CHROMA_DIR) / "properties.json"
_properties_data: List[Dict] = []
_properties_lock = threading.RLock()

def _load_properties():
    global _properties_data
    with _properties_lock:
        if _properties_file.exists():
            try:
                with open(_properties_file, "r", encoding="utf-8") as f:
                    _properties_data = json.load(f)
            except Exception as e:
                print(f"[retrieval] WARN load properties.json: {e}")
                _properties_data = []

def _save_properties():
    with _properties_lock:
        try:
            _properties_file.parent.mkdir(parents=True, exist_ok=True)
            with open(_properties_file, "w", encoding="utf-8") as f:
                json.dump(_properties_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[retrieval] WARN save properties.json: {e}")

_load_properties()

def build_doc_text(p: Dict) -> str:
    am = ", ".join(sorted(p.get("amenities", []) or []))
    city = p.get("city", "") or p.get("location", "")
    country = p.get("country", "")
    desc = p.get("description", "")
    title = p.get("title", "")
    beds = p.get("bedrooms", "")
    price = p.get("price_per_night", "")
    return (
        f"{title}. {desc} Amenities: {am}. "
        f"Beds: {beds}. Price per night: {price}. "
        f"Location: {city}, {country}."
    )

# ---------- Vector mode (Chroma + HF) ----------
_chroma_client = None
_collection = None
_embedder = None

def _init_vector_mode() -> bool:
    """Initialize Chroma persistent client, collection, and embedder."""
    global _chroma_client, _collection, _embedder, _HAS_VECTOR
    if not _HAS_VECTOR:
        return False
    try:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(
                path=CHROMA_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
        if _collection is None:
            # Use a sentence-transformer embedder manually
            _collection = _chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        if _embedder is None:
            _embedder = SentenceTransformer(EMBED_MODEL)
        return True
    except Exception as e:
        print(f"[retrieval] Disable vector mode due to init error: {e}")
        _HAS_VECTOR = False
        return False

def _embed_texts(texts: List[str]) -> List[List[float]]:
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    # SentenceTransformer returns numpy array; convert to python lists for Chroma
    embs = _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

# ---------- Public API ----------
def upsert_property(p: Dict) -> None:
    """
    Upsert a single property. Expects a stable 'id'.
    Works in vector mode if available; otherwise JSON fallback.
    """
    assert "id" in p, "property needs 'id'"
    prop = {
        "id": str(p["id"]),
        "property_id": str(p["id"]),
        "title": p.get("title"),
        "city": p.get("city") or p.get("location"),
        "country": p.get("country"),
        "price_per_night": p.get("price_per_night"),
        "bedrooms": p.get("bedrooms"),
        "amenities": p.get("amenities", []) or [],
        "description": p.get("description", ""),
        "document": build_doc_text(p),
    }

    if _init_vector_mode():
        try:
            # Delete old if exists
            try:
                _collection.delete(ids=[prop["id"]])
            except Exception:
                pass
            _collection.add(
                ids=[prop["id"]],
                documents=[prop["document"]],
                embeddings=_embed_texts([prop["document"]]),
                metadatas=[{
                    "title": prop["title"],
                    "city": prop["city"],
                    "country": prop["country"],
                    "price_per_night": prop["price_per_night"],
                    "bedrooms": prop["bedrooms"],
                    "amenities": json.dumps(prop["amenities"]),
                }],
            )
            return
        except Exception as e:
            print(f"[retrieval] vector upsert error (fallback to JSON): {e}")

    # Fallback path
    with _properties_lock:
        _properties_data[:] = [x for x in _properties_data if str(x.get("id")) != prop["id"]]
        _properties_data.append(prop)
        _save_properties()

def bulk_upsert(properties: List[Dict]) -> None:
    if not properties:
        return
    if _init_vector_mode():
        try:
            ids, docs, metas = [], [], []
            for p in properties:
                pid = str(p["id"])
                doc = build_doc_text(p)
                ids.append(pid)
                docs.append(doc)
                metas.append({
                    "title": p.get("title"),
                    "city": p.get("city") or p.get("location"),
                    "country": p.get("country"),
                    "price_per_night": p.get("price_per_night"),
                    "bedrooms": p.get("bedrooms"),
                    "amenities": json.dumps(p.get("amenities", []) or []),
                })
            # Delete duplicates first to keep Chroma clean
            try:
                _collection.delete(ids=[str(p["id"]) for p in properties])
            except Exception:
                pass
            _collection.add(
                ids=ids,
                documents=docs,
                embeddings=_embed_texts(docs),
                metadatas=metas,
            )
            return
        except Exception as e:
            print(f"[retrieval] vector bulk_upsert error (fallback to JSON): {e}")

    # Fallback: JSON append with de-dupe
    with _properties_lock:
        incoming_ids = {str(p["id"]) for p in properties}
        _properties_data[:] = [x for x in _properties_data if str(x.get("id")) not in incoming_ids]
        for p in properties:
            _properties_data.append({
                "id": str(p["id"]),
                "property_id": str(p["id"]),
                "title": p.get("title"),
                "city": p.get("city") or p.get("location"),
                "country": p.get("country"),
                "price_per_night": p.get("price_per_night"),
                "bedrooms": p.get("bedrooms"),
                "amenities": p.get("amenities", []) or [],
                "description": p.get("description", ""),
                "document": build_doc_text(p),
            })
        _save_properties()

def query_properties(
    query_text: str,
    k: int = 10,
    budget: Optional[float] = None,
    city: Optional[str] = None,
    beds: Optional[int] = None,
    amenities: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Retrieve top-k properties. Uses Chroma+HF if available, otherwise JSON text search.
    Filters (budget/city/beds/amenities) are applied on metadata after retrieval.
    Returns up to 5 compact dicts with basic fields + snippet + score.
    """
    # ---------- Vector mode ----------
    if _init_vector_mode():
        try:
            q_emb = _embed_texts([query_text])[0]
            # Initial ANN search
            res = _collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                include=["metadatas", "documents", "distances"],
            )
            items = []
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]

            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                pid = meta.get("id", f"property_{i}")  # Use id from metadata or generate one
                # Convert amenities back from JSON string
                try:
                    am = json.loads(meta.get("amenities") or "[]")
                except Exception:
                    am = meta.get("amenities") or []

                row = {
                    "id": pid,
                    "title": meta.get("title"),
                    "city": meta.get("city"),
                    "country": meta.get("country"),
                    "price_per_night": meta.get("price_per_night"),
                    "bedrooms": meta.get("bedrooms"),
                    "amenities": am,
                    "snippet": (doc or "")[:240] + ("..." if doc and len(doc) > 240 else ""),
                    # smaller distance = closer → convert to score ~ 1 - normalized distance
                    "score": float(max(0.0, 1.0 - (dist or 0.0))),
                }
                items.append(row)

            # Apply filters post-retrieval
            items = _apply_filters(items, budget=budget, city=city, beds=beds, amenities=amenities)
            # Sort by score descending & truncate to 5
            items.sort(key=lambda x: x.get("score", 0), reverse=True)
            return items[:5]
        except Exception as e:
            print(f"[retrieval] vector query error (fallback to JSON): {e}")

    # ---------- Fallback: JSON text search ----------
    with _properties_lock:
        snapshot = list(_properties_data)
    filtered = []
    for prop in snapshot:
        filtered.append({
            "id": prop.get("property_id"),
            "title": prop.get("title"),
            "city": prop.get("city"),
            "country": prop.get("country"),
            "price_per_night": prop.get("price_per_night"),
            "bedrooms": prop.get("bedrooms"),
            "amenities": prop.get("amenities", []),
            "document": prop.get("document", ""),
        })

    # crude scoring
    q = (query_text or "").lower()
    scored: List[Tuple[float, Dict]] = []
    for p in filtered:
        score = 0.0
        doc = p.get("document", "") or ""
        title = p.get("title", "") or ""
        if q and q in doc.lower():
            score += 1.0
        if q and q in title.lower():
            score += 2.0
        for w in q.split():
            if w in doc.lower():
                score += 0.5
            if w in title.lower():
                score += 1.0
        if score > 0.0:
            p2 = p.copy()
            p2["snippet"] = doc[:240] + ("..." if len(doc) > 240 else "")
            p2["score"] = score
            scored.append((score, p2))

    # apply filters and sort
    out = [p for _, p in sorted(scored, key=lambda x: x[0], reverse=True)]
    out = _apply_filters(out, budget=budget, city=city, beds=beds, amenities=amenities)
    return out[:5]

# ---------- Helpers ----------
# In query_properties function, update the filter logic:

def _apply_filters(items: List[Dict],
                   budget: Optional[float],
                   city: Optional[str],
                   beds: Optional[int],
                   amenities: Optional[List[str]]) -> List[Dict]:
    res = []
    for p in items:
        # Budget filter
        if budget is not None:
            price = float(p.get("price_per_night") or 0)
            if price > budget:
                continue
        
        # City filter - more flexible matching
        if city:
            city_lower = city.lower().strip()
            prop_city = (p.get("city") or "").lower().strip()
            # Check if the search city is contained in property city
            if city_lower not in prop_city and prop_city not in city_lower:
                continue
        
        # Beds filter
        if beds:
            prop_beds = int(p.get("bedrooms") or 0)
            if prop_beds < beds:
                continue
        
        # Amenities filter
        if amenities:
            prop_amenities = [a.lower() for a in (p.get("amenities") or [])]
            required_amenities = [a.lower() for a in amenities]
            if not all(a in prop_amenities for a in required_amenities):
                continue
        
        res.append(p)
    return res
