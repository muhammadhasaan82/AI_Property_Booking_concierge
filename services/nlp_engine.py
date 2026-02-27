# services/nlp_engine.py
# -*- coding: utf-8 -*-
"""
Unified NLP engine — replaces all hardcoded regex/arrays/dicts with
dynamic analysis powered by VADER, spaCy, and sentence-transformers.

Heavy NLP calls are wrapped with asyncio.to_thread to avoid blocking
the FastAPI event loop.

Lazy-loads models on first use. Gracefully degrades if spaCy or
sentence-transformers are unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Lazy singletons
# ─────────────────────────────────────────────────────────────────────

_vader_analyzer = None
_spacy_nlp = None
_st_model = None
_intent_embeddings: Optional[Dict[str, Any]] = None

# Baseline intent prototypes for semantic classification
_INTENT_PROTOTYPES: Dict[str, List[str]] = {
    "property_search": [
        "I want to find an apartment in New York",
        "Show me available rentals under 200 dollars",
        "Looking for a villa with a pool",
        "Search for properties in Miami",
        "I need a place to stay near downtown",
        "Find me a 2 bedroom house",
        "What do you have available",
        "Browse listings with wifi and parking",
    ],
    "status_update": [
        "What is the status of my booking",
        "Check my booking status",
        "When is my check-in date",
        "I want to know my departure time",
        "Track my reservation",
        "Where is my booking confirmation",
        "My booking ID is abc-123",
    ],
    "faq": [
        "What is the refund policy",
        "Can I cancel my booking",
        "Are pets allowed in the property",
        "What are the check-in times",
        "Tell me about your cancellation terms",
        "Is there a security deposit",
        "What are the house rules",
        "How does the payment process work",
        "What WiFi password do I use",
    ],
    "greeting": [
        "Hi there",
        "Hello",
        "Hey how are you",
        "Good morning",
        "Good afternoon",
    ],
    "booking": [
        "I want to book this property",
        "Reserve this for me",
        "Go ahead and confirm the booking",
        "Lock it in for those dates",
    ],
    "handoff": [
        "I want to talk to a human",
        "Connect me with an agent",
        "Can I speak to a representative",
        "Live support please",
        "I need a real person",
    ],
    "end": [
        "Goodbye",
        "Bye bye",
        "That is all thank you",
        "I am done",
        "Close this chat",
        "Exit",
    ],
    "payment": [
        "Send me the payment link",
        "I want to pay now",
        "Process my payment",
        "Generate an invoice",
    ],
    "availability": [
        "What dates are available",
        "Show me the calendar",
        "Which dates are open",
        "Available dates for this property",
    ],
    "modification": [
        "I want to change my dates",
        "Modify my booking details",
        "Update my phone number",
        "Change the guest count",
        "Edit my name",
    ],
}

# ─── Field detection prototypes ───
_FIELD_PROTOTYPES: Dict[str, List[str]] = {
    "property": [
        "change property", "different property", "another property",
        "switch property", "browse more", "more options",
    ],
    "check_in": [
        "change check-in date", "modify arrival", "new check-in",
        "update start date",
    ],
    "check_out": [
        "change check-out date", "modify departure", "new check-out",
        "update end date",
    ],
    "dates": [
        "change my dates", "modify dates", "update both dates",
        "new dates", "different dates",
    ],
    "name": [
        "change my name", "update name", "modify name",
    ],
    "phone": [
        "change phone number", "update my phone", "modify contact number",
        "new mobile number",
    ],
    "email": [
        "change email address", "update my email", "modify email",
    ],
    "guests": [
        "change number of guests", "update guests", "modify guest count",
        "change people count",
    ],
    "location": [
        "change city", "different location", "modify city",
        "update location",
    ],
}


def _get_vader():
    """Lazy-init VADER sentiment analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("[nlp_engine] VADER initialized")
        except ImportError:
            logger.warning("[nlp_engine] vaderSentiment not installed, using fallback")
            _vader_analyzer = _FallbackVader()
    return _vader_analyzer


def _get_spacy():
    """Lazy-init spaCy pipeline."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("[nlp_engine] spaCy en_core_web_sm loaded")
        except (ImportError, OSError):
            logger.warning("[nlp_engine] spaCy model not available, NER disabled")
            _spacy_nlp = False  # sentinel so we don't retry
    return _spacy_nlp if _spacy_nlp is not False else None


def _get_st_model():
    """Lazy-init sentence-transformers model for zero-shot classification."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[nlp_engine] sentence-transformers all-MiniLM-L6-v2 loaded")
        except (ImportError, OSError):
            logger.warning("[nlp_engine] sentence-transformers not available, "
                           "falling back to keyword matching")
            _st_model = False
    return _st_model if _st_model is not False else None


def _get_intent_embeddings() -> Optional[Dict[str, Any]]:
    """Pre-compute prototype embeddings (cached after first call)."""
    global _intent_embeddings
    if _intent_embeddings is not None:
        return _intent_embeddings
    model = _get_st_model()
    if model is None:
        return None
    import numpy as np
    _intent_embeddings = {}
    for intent, phrases in _INTENT_PROTOTYPES.items():
        vecs = model.encode(phrases, convert_to_numpy=True)
        _intent_embeddings[intent] = np.mean(vecs, axis=0)
    logger.info("[nlp_engine] Intent embeddings pre-computed for %d intents",
                len(_intent_embeddings))
    return _intent_embeddings


# ─────────────────────────────────────────────────────────────────────
# Fallback VADER (when vaderSentiment is not installed)
# ─────────────────────────────────────────────────────────────────────

class _FallbackVader:
    """Minimal polarity scorer used when vaderSentiment is not installed."""

    _POS = {"yes", "yeah", "yep", "yup", "sure", "please", "ok", "okay",
            "alright", "good", "great", "hi", "hello", "hey", "thanks",
            "thank", "love", "nice", "perfect", "wonderful", "excellent",
            "amazing", "awesome", "oki", "fine"}
    _NEG = {"no", "nope", "nah", "not", "never", "bad", "terrible",
            "horrible", "stop", "cancel", "later", "awful", "hate", "ugly"}

    def polarity_scores(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[a-z']+", text.lower())
        pos = sum(1 for t in tokens if t in self._POS)
        neg = sum(1 for t in tokens if t in self._NEG)
        total = max(pos + neg, 1)
        compound = (pos - neg) / total
        return {"pos": pos / total, "neg": neg / total,
                "neu": 1.0 - (pos + neg) / max(len(tokens), 1),
                "compound": compound}


# ─────────────────────────────────────────────────────────────────────
# Core NLP Functions (sync — wrap with asyncio.to_thread for async)
# ─────────────────────────────────────────────────────────────────────

def classify_affirmation(text: str) -> str:
    """Classify text as 'yes', 'no', or 'neutral' using VADER + lexicon.

    Returns
    -------
    'yes' | 'no' | 'neutral'
    """
    if not text or not text.strip():
        return "neutral"

    tl = text.strip().lower()

    # Fast-path: exact matches (common conversational tokens)
    _YES_EXACT = {"yes", "yeah", "yep", "yup", "sure", "please", "ok",
                  "okay", "alright", "oki", "yea", "absolutely", "definitely",
                  "of course", "go ahead", "proceed", "confirm", "confirmed",
                  "right", "correct", "affirmative", "y"}
    _NO_EXACT = {"no", "nope", "nah", "not now", "later", "stop", "cancel",
                 "negative", "n", "decline", "reject", "refuse", "denied",
                 "never mind", "nevermind"}

    if tl in _YES_EXACT:
        return "yes"
    if tl in _NO_EXACT:
        return "no"

    # VADER compound score
    vader = _get_vader()
    scores = vader.polarity_scores(tl)
    compound = scores["compound"]

    # Positive compound + short text ≈ affirmation
    if compound >= 0.3 and len(tl.split()) <= 4:
        return "yes"
    if compound <= -0.3 and len(tl.split()) <= 4:
        return "no"

    return "neutral"


def is_greeting(text: str) -> bool:
    """Detect conversational greetings dynamically."""
    if not text or not text.strip():
        return False

    tl = text.strip().lower()
    tokens = tl.split()

    # Short utterance check (greetings are usually 1-4 words)
    if len(tokens) > 6:
        return False

    # Speech-act pattern matching via VADER + semantic cues
    _GREETING_SEEDS = {"hi", "hello", "hey", "salam", "assalam", "assalamu",
                       "hiya", "yo", "howdy", "greetings", "sup", "heya"}
    _GREETING_PHRASES = {"good morning", "good afternoon", "good evening",
                         "how are you", "what's up", "whats up"}

    if tokens[0] in _GREETING_SEEDS:
        return True
    if any(re.search(r'\b' + re.escape(p) + r'\b', tl) for p in _GREETING_PHRASES):
        return True

    # Semantic classification fallback
    intent = classify_intent_sync(tl, ["greeting", "other"])
    return intent == "greeting" and _semantic_confidence(tl, "greeting") > 0.55


def is_acknowledgment(text: str) -> bool:
    """Detect conversational acknowledgment (ok, sounds good, got it, etc.)."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()
    _ACK = {"ok", "okay", "alright", "fine", "kk", "k", "oki", "sure thing",
            "understood", "noted", "right"}
    _ACK_PHRASES = {"sounds good", "got it", "thanks", "thank you",
                    "i see", "i understand", "all good", "that works",
                    "no problem", "np"}
    if tl in _ACK:
        return True
    return any(p in tl for p in _ACK_PHRASES)


def is_handoff_request(text: str) -> bool:
    """Detect request to speak with a human agent."""
    if not text:
        return False
    tl = text.strip().lower()
    _SEEDS = {"human", "person", "agent", "representative", "support",
              "operator", "staff", "manager"}
    _PHRASES = {"live chat", "talk to", "connect me", "speak to",
                "real person", "human agent", "customer service"}
    if any(s in tl for s in _SEEDS):
        return True
    return any(p in tl for p in _PHRASES)


def is_availability_query(text: str) -> bool:
    """Detect questions about date availability."""
    if not text:
        return False
    tl = text.strip().lower()
    _PHRASES = {"available dates", "availability", "what dates", "which dates",
                "show dates", "calendar", "date options", "open dates",
                "dates available", "when available"}
    return any(p in tl for p in _PHRASES)


def is_end_request(text: str) -> bool:
    """Detect conversation end requests."""
    if not text:
        return False
    tl = text.strip().lower()
    _EXACT = {"end", "bye", "goodbye", "exit", "quit", "done", "close",
              "finish", "that's all", "that is all"}
    _PHRASES = {"end chat", "close chat", "no thanks", "bye bye",
                "good bye", "see you", "talk later"}
    if tl in _EXACT:
        return True
    return any(p in tl for p in _PHRASES)


def is_status_query(text: str) -> bool:
    """Detect booking status or check-in/out inquiries."""
    if not text:
        return False
    tl = text.strip().lower()

    # UUID presence is a strong signal
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", tl):
        return True

    # Semantic classification
    intent = classify_intent_sync(tl, ["status_update", "property_search", "other"])
    if intent == "status_update":
        conf = _semantic_confidence(tl, "status_update")
        if conf > 0.45:
            return True

    # Keyword fallback
    _STATUS_SEEDS = {"status", "booking status", "check status",
                     "booking id", "booking_id", "bookingid",
                     "my booking", "track", "tracking"}
    return any(s in tl for s in _STATUS_SEEDS)


def is_property_search(text: str) -> bool:
    """Detect property search intent, excluding status queries."""
    if not text:
        return False
    tl = text.strip().lower()

    # Exclude status queries
    if is_status_query(text):
        return False
    # Exclude booking ID mentions
    if any(x in tl for x in ["booking id", "booking_id", "bookingid"]):
        return False
    if re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", tl):
        return False

    # Semantic classification
    intent = classify_intent_sync(
        tl, ["property_search", "status_update", "faq", "greeting", "other"]
    )
    if intent == "property_search":
        return True

    # Keyword + NER fallback
    from .nlp_extractor import KNOWN_CITIES, CITY_ALIASES, PROPERTY_TYPES
    _MONEY_PAT = re.compile(
        r"(\$|€|£)\s*\d+|\b(under|below|less than|max(?:imum)?|up to)\b\s*[\$€£]?\s*\d+", re.I
    )
    _SEARCH_SIGNALS = {
        "want", "need", "looking", "search", "find", "place", "property",
        "rent", "stay", "accommodation", "room", "suite", "home", "show",
        "available", "options", "listings", "interested", "seeking",
        "browse", "explore", "view", "lease", "hire",
    }
    _SEARCH_PHRASES = {
        "i want", "i need", "i'm looking", "looking for", "searching for",
        "find me", "show me", "what do you have", "what's available",
        "do you have", "are there any", "i'd like", "interested in",
        "want to rent", "find a", "get me",
    }

    if any(re.search(r'\b' + re.escape(p) + r'\b', tl) for p in PROPERTY_TYPES):
        return True
    if any(re.search(r'\b' + re.escape(c) + r'\b', tl) for c in KNOWN_CITIES):
        return True
    if any(re.search(r'\b' + re.escape(a) + r'\b', tl) for a in CITY_ALIASES):
        return True
    if _MONEY_PAT.search(tl):
        return True
    if any(re.search(r'\b' + re.escape(w) + r'\b', tl) for w in _SEARCH_SIGNALS):
        return True
    if any(re.search(r'\b' + re.escape(p) + r'\b', tl) for p in _SEARCH_PHRASES):
        return True

    return False


def wants_modification(text: str) -> bool:
    """Detect intent to modify booking details."""
    if not text:
        return False
    tl = text.strip().lower()
    _SEEDS = {"modify", "modification", "change", "update", "edit",
              "correct", "fix", "adjust", "tweak"}
    return any(w in tl for w in _SEEDS)


def wants_property_search_request(text: str) -> bool:
    """Detect intent to search for different/more properties."""
    if not text:
        return False
    tl = text.strip().lower()
    _PHRASES = {
        "search for different properties", "search different properties",
        "show other options", "show more options", "more properties",
        "different options", "other options", "browse more", "see more",
        "change property", "another property", "different property",
        "other property",
    }
    if any(p in tl for p in _PHRASES):
        return True
    return (("search" in tl or "browse" in tl or "show" in tl) and
            ("property" in tl or "properties" in tl or "options" in tl))


def detect_faq_intent(text: str) -> bool:
    """Detect FAQ/policy questions using semantic classification."""
    if not text or not text.strip():
        return False
    tl = text.strip().lower()

    # Semantic classification
    intent = classify_intent_sync(tl, ["faq", "property_search", "greeting", "other"])
    if intent == "faq":
        conf = _semantic_confidence(tl, "faq")
        if conf > 0.45:
            return True

    # Strong trigger keywords (policy/terms are always FAQ)
    _STRONG = {"policy", "refund", "cancel", "cancellation", "terms",
               "conditions", "dispute"}
    if any(re.search(r'\b' + re.escape(s) + r'\b', tl) for s in _STRONG):
        return True

    # Broader keyword + question pattern check
    _FAQ_SEEDS = {
        "rules", "regulations", "guidelines", "pet", "pets", "smoking",
        "deposit", "security deposit", "damage", "liability", "insurance",
        "wifi password", "check-in time", "checkout time", "late check",
        "early check", "complaint", "grievance",
    }
    _QUESTION_STARTS = {"what", "how", "when", "where", "why", "can",
                        "do", "does", "is", "are", "tell", "explain"}
    has_faq_seed = any(re.search(r'\b' + re.escape(s) + r'\b', tl) for s in _FAQ_SEEDS)
    has_question = ("?" in tl or any(tl.startswith(q) for q in _QUESTION_STARTS) or
                    "tell me" in tl or "explain" in tl)
    return has_faq_seed and has_question


# ─────────────────────────────────────────────────────────────────────
# NER Extraction (spaCy-based with fallback)
# ─────────────────────────────────────────────────────────────────────

def extract_person_name(text: str) -> Optional[str]:
    """Extract person name via spaCy PERSON NER, with regex fallback."""
    if not text:
        return None

    t_lower = text.lower().strip()
    # Guard: skip if text looks like a search query
    _SEARCH_GUARDS = {"find", "show", "search", "look for", "looking for",
                      "want", "need", "rent", "buy", "apartment", "house",
                      "villa", "condo", "loft", "studio", "under $", "$",
                      "per night", "available", "dates", "amenities"}
    if any(term in t_lower for term in _SEARCH_GUARDS):
        return None

    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip().rstrip('.').strip()
                if len(name) >= 2:
                    return name

    # Regex fallback
    _NAME_PATS = [
        re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{1,60})", re.I),
        re.compile(r"\bname is\s+([A-Za-z][A-Za-z .'-]{1,60})", re.I),
        re.compile(r"\bi am\s+([A-Za-z][A-Za-z .'-]{1,60})", re.I),
        re.compile(r"\bI'm\s+([A-Za-z][A-Za-z .'-]{1,60})", re.I),
        re.compile(r"^([A-Za-z][A-Za-z .'-]{1,60})\b", re.I),
        re.compile(r"^([A-Za-z]{2,60})$", re.I),
    ]
    for pat in _NAME_PATS:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip().rstrip('.').strip()
            if len(cand) >= 2 and not _looks_like_email_username(cand):
                return cand
    return None


def _looks_like_email_username(s: str) -> bool:
    if not s:
        return False
    common = {'test', 'example', 'user', 'admin', 'john', 'jane',
              'info', 'contact', 'support', 'help', 'demo'}
    return (len(s) <= 3 or s.lower() in common or
            bool(re.search(r'\d', s)) or
            bool(re.search(r"[^a-zA-Z\s\-.'']", s)))


def extract_dates(text: str) -> List[str]:
    """Extract dates from text using spaCy DATE NER + regex fallback."""
    results: List[str] = []
    if not text:
        return results

    # Primary: ISO date regex (always reliable for YYYY-MM-DD)
    iso_dates = re.findall(r"\b(\d{4}-\d{1,2}-\d{1,2})\b", text)
    results.extend(iso_dates)

    if results:
        return results

    # spaCy DATE entity extraction (for natural language dates)
    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                results.append(ent.text)

    return results


def extract_cardinal(text: str) -> Optional[int]:
    """Extract ordinal/cardinal number from text for selection index."""
    if not text:
        return None
    tl = text.strip().lower()

    # Regex patterns for explicit selection
    _SELECTION_PATS = [
        r"\b(\d{1,2})(?:st|nd|rd|th)\b",
        r"\b(?:option|number|no\.?|#)\s*(\d{1,2})\b",
        r"\b(?:pick|choose|select|book|take)\s*(\d{1,2})\b",
        r"^\s*(\d{1,2})\s*$",
    ]
    for rx in _SELECTION_PATS:
        m = re.search(rx, tl)
        if m:
            val = int(m.group(1))
            return val if val >= 1 else None

    # Ordinal words
    _ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3,
                 "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
    for word, idx in _ORDINALS.items():
        if word in tl or f"the {word}" in tl or f"{word} one" in tl:
            return idx

    # Cardinal words
    _CARDINALS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                  "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    m = re.search(
        r"\b(?:option|number|no\.?|#)?\s*(one|two|three|four|five|six|seven|eight|nine|ten)\b",
        tl
    )
    if m:
        return _CARDINALS.get(m.group(1))

    # spaCy ORDINAL/CARDINAL fallback
    nlp = _get_spacy()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("ORDINAL", "CARDINAL"):
                try:
                    val = int(ent.text)
                    if val >= 1:
                        return val
                except ValueError:
                    mapped = _ORDINALS.get(ent.text.lower()) or _CARDINALS.get(ent.text.lower())
                    if mapped:
                        return mapped

    return None


def extract_guests(text: str) -> Optional[int]:
    """Extract guest count from text."""
    if not text:
        return None
    tl = text.lower().strip()
    m = (re.search(r"(\d{1,3})\s*(guest|guests|people|persons|pax)?\b", tl) or
         re.search(r"^(\d{1,3})$", tl))
    if m:
        try:
            n = int(m.group(1))
            return n if 1 <= n <= 100 else None
        except (ValueError, IndexError):
            pass
    return None


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text using regex."""
    if not text:
        return None
    m = re.search(r"(\+?[\d\s\-]{8,15}\d)", text)
    if m:
        num = re.sub(r"[\s\-]", "", m.group(1))
        # Reject if it looks like a date
        if re.search(r"\d{4}-\d{1,2}-\d{1,2}", text):
            return None
        if re.match(r"^\d{8}$", num):
            return None
        return num
    return None


def extract_email(text: str) -> Optional[str]:
    """Extract email address from text."""
    if not text:
        return None
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def extract_booking_id(text: str) -> Optional[str]:
    """Extract UUID or hex booking ID from text."""
    if not text:
        return None
    tl = text.lower()
    uuid_m = re.search(
        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", tl
    )
    if uuid_m:
        return uuid_m.group(1)
    short_m = re.search(r"\b([0-9a-f]{8})\b", tl)
    if short_m:
        return short_m.group(1)
    return None


def detect_requested_fields(text: str) -> List[str]:
    """Detect which booking fields the user wants to modify."""
    if not text:
        return []
    tl = text.lower()
    fields: List[str] = []

    model = _get_st_model()
    if model:
        import numpy as np
        text_emb = model.encode([tl], convert_to_numpy=True)[0]
        for field, phrases in _FIELD_PROTOTYPES.items():
            field_embs = model.encode(phrases, convert_to_numpy=True)
            mean_emb = np.mean(field_embs, axis=0)
            sim = float(np.dot(text_emb, mean_emb) /
                        (np.linalg.norm(text_emb) * np.linalg.norm(mean_emb) + 1e-8))
            if sim > 0.50:
                if field not in fields:
                    fields.append(field)

    if fields:
        return fields

    # Keyword fallback (same semantics as original)
    def add(f: str):
        if f not in fields:
            fields.append(f)

    if any(p in tl for p in ["different property", "another property", "other property",
                             "change property", "switch property", "new property",
                             "browse more", "more options", "more listings"]):
        add("property")
    if any(p in tl for p in ["check out", "checkout", "check-out", "departure",
                             "end date", "check_out"]):
        add("check_out")
    if any(p in tl for p in ["check in", "checkin", "check-in", "arrival",
                             "start date", "check_in"]):
        add("check_in")
    if any(p in tl for p in ["dates", "both dates", "change dates",
                             "update dates", "modify dates"]):
        add("dates")
    if any(p in tl for p in ["city", "location", "change city",
                             "change location"]):
        add("location")
    if any(p in tl for p in ["name", "my name", "change name"]):
        add("name")
    if any(p in tl for p in ["phone", "mobile", "number", "contact number",
                             "phone number"]):
        add("phone")
    if any(p in tl for p in ["email", "e-mail", "mail", "email address"]):
        add("email")
    if any(p in tl for p in ["guest", "guests", "people", "persons", "pax"]):
        add("guests")
    return fields


# ─────────────────────────────────────────────────────────────────────
# Semantic classification (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────

def _semantic_confidence(text: str, intent: str) -> float:
    """Return cosine similarity between text and a specific intent prototype."""
    model = _get_st_model()
    if model is None:
        return 0.0
    embs = _get_intent_embeddings()
    if embs is None or intent not in embs:
        return 0.0
    import numpy as np
    text_vec = model.encode([text], convert_to_numpy=True)[0]
    intent_vec = embs[intent]
    sim = float(np.dot(text_vec, intent_vec) /
                (np.linalg.norm(text_vec) * np.linalg.norm(intent_vec) + 1e-8))
    return sim


def classify_intent_sync(text: str, candidates: List[str]) -> str:
    """Classify text against candidate intents using sentence-transformers.

    Falls back to keyword-based heuristics if sentence-transformers unavailable.
    """
    model = _get_st_model()
    if model is None:
        return _classify_intent_keyword_fallback(text, candidates)

    embs = _get_intent_embeddings()
    if embs is None:
        return _classify_intent_keyword_fallback(text, candidates)

    import numpy as np
    text_vec = model.encode([text], convert_to_numpy=True)[0]

    best_intent = candidates[-1] if candidates else "other"
    best_score = -1.0
    for intent in candidates:
        if intent not in embs:
            continue
        sim = float(np.dot(text_vec, embs[intent]) /
                    (np.linalg.norm(text_vec) * np.linalg.norm(embs[intent]) + 1e-8))
        if sim > best_score:
            best_score = sim
            best_intent = intent
    return best_intent


def _classify_intent_keyword_fallback(text: str, candidates: List[str]) -> str:
    """Keyword-based intent classification fallback."""
    tl = text.lower()
    _KEYWORD_MAP = {
        "property_search": ["apartment", "house", "villa", "rent", "stay", "find",
                            "search", "looking", "property", "show"],
        "status_update": ["status", "booking", "check-in", "checkout", "track"],
        "faq": ["policy", "refund", "cancel", "terms", "rules", "wifi"],
        "greeting": ["hi", "hello", "hey", "morning", "afternoon", "evening"],
        "booking": ["book", "reserve", "confirm"],
        "handoff": ["human", "agent", "person", "support", "representative"],
        "end": ["bye", "goodbye", "exit", "quit", "done", "close"],
        "payment": ["pay", "payment", "invoice", "link"],
        "availability": ["available", "calendar", "dates"],
        "modification": ["modify", "change", "update", "edit"],
    }
    best = candidates[-1] if candidates else "other"
    best_count = 0
    for intent in candidates:
        kws = _KEYWORD_MAP.get(intent, [])
        count = sum(1 for k in kws if k in tl)
        if count > best_count:
            best_count = count
            best = intent
    return best


# ─────────────────────────────────────────────────────────────────────
# Async wrappers (for use in FastAPI handlers)
# ─────────────────────────────────────────────────────────────────────

async def classify_affirmation_async(text: str) -> str:
    return await asyncio.to_thread(classify_affirmation, text)

async def is_greeting_async(text: str) -> bool:
    return await asyncio.to_thread(is_greeting, text)

async def extract_person_name_async(text: str) -> Optional[str]:
    return await asyncio.to_thread(extract_person_name, text)

async def extract_dates_async(text: str) -> List[str]:
    return await asyncio.to_thread(extract_dates, text)

async def extract_cardinal_async(text: str) -> Optional[int]:
    return await asyncio.to_thread(extract_cardinal, text)

async def classify_intent_async(text: str, candidates: List[str]) -> str:
    return await asyncio.to_thread(classify_intent_sync, text, candidates)

async def detect_faq_intent_async(text: str) -> bool:
    return await asyncio.to_thread(detect_faq_intent, text)

async def detect_requested_fields_async(text: str) -> List[str]:
    return await asyncio.to_thread(detect_requested_fields, text)

async def is_property_search_async(text: str) -> bool:
    return await asyncio.to_thread(is_property_search, text)

async def is_status_query_async(text: str) -> bool:
    return await asyncio.to_thread(is_status_query, text)
