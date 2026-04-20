from __future__ import annotations
from contextlib import contextmanager
from huggingface_hub import login, whoami
import os
from dotenv import load_dotenv
import json
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from huggingface_hub import login, whoami
from app.services.dynamic_config import get_retrieval_config

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
_RETRIEVAL_CFG = get_retrieval_config()
CHROMA_DIR = os.getenv("CHROMA_DIR", _RETRIEVAL_CFG.chroma.persist_dir)
EMBED_MODEL = os.getenv("EMBED_MODEL", _RETRIEVAL_CFG.embeddings.model_name)
EMBED_NORMALIZE = bool(_RETRIEVAL_CFG.embeddings.normalize_embeddings)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", _RETRIEVAL_CFG.chroma.collection_name)
RAG_LOCAL_MODELS_ONLY = os.getenv("RAG_LOCAL_MODELS_ONLY", "1").lower() not in {"0", "false", "no"}


@contextmanager
def _local_model_load(enabled: bool):
    if not enabled:
        yield
        return

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _is_local_model_reference(model_name: str) -> bool:
    try:
        return Path(model_name).expanduser().exists()
    except OSError:
        return False


_HAS_VECTOR = True
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except Exception as _e:
    print(f"[retrieval] Vector mode unavailable ({_e}); using JSON fallback.")
    _HAS_VECTOR = False


_properties_file = Path(CHROMA_DIR) / "properties.json"
_properties_data: List[Dict] = []
_properties_lock = threading.RLock()


def _retrieval_runtime():
    return get_retrieval_config().retrieval

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


_chroma_client = None
_collection = None
_embedder = None
login(token=os.getenv("HF_TOKEN"))
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
            login(token=os.getenv("HF_TOKEN"))
            _collection = _chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        if _embedder is None:
            if RAG_LOCAL_MODELS_ONLY and not _is_local_model_reference(EMBED_MODEL):
                login(token=os.getenv("HF_TOKEN"))
                raise RuntimeError(
                    f"Embedding model '{EMBED_MODEL}' is not local. "
                    "Set RAG_LOCAL_MODELS_ONLY=0 to allow remote model downloads."
                )
            with _local_model_load(RAG_LOCAL_MODELS_ONLY):
                cache_folder = os.getenv("cache_folder")
                _embedder = SentenceTransformer(EMBED_MODEL, cache_folder=cache_folder)
        return True
    except Exception as e:
        print(f"[retrieval] Disable vector mode due to init error: {e}")
        _HAS_VECTOR = False
        return False

def _embed_texts(texts: List[str]) -> List[List[float]]:
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    embs = _embedder.encode(texts, normalize_embeddings=EMBED_NORMALIZE, show_progress_bar=False)
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]


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
    k: Optional[int] = None,
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
    runtime = _retrieval_runtime()
    if k is None:
        k = runtime.top_k


    if _init_vector_mode():
        try:
            q_emb = _embed_texts([query_text])[0]

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
                pid = meta.get("id", f"property_{i}")
 
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

                    "score": float(max(0.0, 1.0 - (dist or 0.0))),
                }
                items.append(row)


            items = _apply_filters(items, budget=budget, city=city, beds=beds, amenities=amenities)

            items.sort(key=lambda x: x.get("score", 0), reverse=True)
            return items[:runtime.result_limit]
        except Exception as e:
            print(f"[retrieval] vector query error (fallback to JSON): {e}")


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

  
    q = (query_text or "").lower()
    weights = runtime.scoring_weights
    scored: List[Tuple[float, Dict]] = []
    for p in filtered:
        score = 0.0
        doc = p.get("document", "") or ""
        title = p.get("title", "") or ""
        if q and q in doc.lower():
            score += weights.exact_doc
        if q and q in title.lower():
            score += weights.exact_title
        for w in q.split():
            if w in doc.lower():
                score += weights.token_doc
            if w in title.lower():
                score += weights.token_title
        if score > 0.0:
            p2 = p.copy()
            p2["snippet"] = doc[:240] + ("..." if len(doc) > 240 else "")
            p2["score"] = score
            scored.append((score, p2))


    out = [p for _, p in sorted(scored, key=lambda x: x[0], reverse=True)]
    out = _apply_filters(out, budget=budget, city=city, beds=beds, amenities=amenities)
    return out[:runtime.result_limit]

def _apply_filters(items: List[Dict],
                   budget: Optional[float],
                   city: Optional[str],
                   beds: Optional[int],
                   amenities: Optional[List[str]]) -> List[Dict]:
    res = []
    for p in items:

        if budget is not None:
            price = float(p.get("price_per_night") or 0)
            if price > budget:
                continue
        

        if city:
            city_lower = city.lower().strip()
            prop_city = (p.get("city") or "").lower().strip()

            if city_lower not in prop_city and prop_city not in city_lower:
                continue
        

        if beds:
            prop_beds = int(p.get("bedrooms") or 0)
            if prop_beds < beds:
                continue
        

        if amenities:
            prop_amenities = [a.lower() for a in (p.get("amenities") or [])]
            required_amenities = [a.lower() for a in amenities]
            if not all(a in prop_amenities for a in required_amenities):
                continue
        
        res.append(p)
    return res

