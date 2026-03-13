import re
import os

AGENTS_PATH = r'c:\Users\ASUS\Desktop\Hotel booking\backend\app\services\agents.py'
NLP_PATH = r'c:\Users\ASUS\Desktop\Hotel booking\backend\app\services\nlp_extractor.py'
TRACING_PATH = r'c:\Users\ASUS\Desktop\Hotel booking\backend\app\services\tracing.py'

print("--- PHASE 1, 2, 3: agents.py ---")
with open(AGENTS_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Make sure imports exist
if 'from app.services.confirmation_helpers import _render_receipt' not in text:
    text = 'from app.services.confirmation_helpers import _render_receipt\n' + text
if 'import app.services.config as config' not in text:
    text = 'import app.services.config as config\n' + text
if 'from .state_keys import SK' not in text:
    text = 'from .state_keys import SK\n' + text

# Phase 1: Wipe monoliths and replace with exact line
# If it's already using confirmation_helpers._render_receipt(persisted), replace it to precisely match user request.
text, c0 = re.subn(r'receipt\s*=\s*confirmation_helpers\._render_receipt\(persisted\)', 'receipt = _render_receipt(persisted)', text)
text, c1 = re.subn(r'receipt\s*=\s*f\"\"\"📋 \*\*BOOKING SUMMARY\*\*(?:.*?)(?<!\\)\"\"\"', 'receipt = _render_receipt(persisted)', text, flags=re.DOTALL)
print(f"Replaced {c0 + c1} receipt generation lines.")

# Phase 2: Eliminate hardcoded arrays
text, c2 = re.subn(r'required\s*=\s*\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*\]', 'required = config.REQUIRED_FIELDS', text)
text, c3 = re.subn(r'required\s*=\s*\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*,\s*"selected_property"\s*\]', 'required = config.REQUIRED_FIELDS + [SK.selected_property]', text)
text, c4 = re.subn(r'required\s*=\s*\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*,\s*SK\.selected_property\s*\]', 'required = config.REQUIRED_FIELDS + [SK.selected_property]', text)

# For any intermediate array notation from my previous work:
text, c5 = re.subn(r'required\s*=\s*\[\*config\.REQUIRED_FIELDS,\s*SK\.selected_property\]', 'required = config.REQUIRED_FIELDS + [SK.selected_property]', text)
# And replace iteration if any:
text, c6 = re.subn(r'for rk in config\.REQUIRED_FIELDS:', 'for rk in config.REQUIRED_FIELDS:', text)

print(f"Replaced {c2+c3+c4+c5+c6} hardcoded arrays.")

# Phase 3: Enforce typed state keys
keys_to_replace = [
    "awaiting_post_mod_choice", "receipt_shown", "awaiting_field", "modifying_dates", 
    "selected_property", "recent_property_id", "recent_selection_index", 
    "awaiting_selection_confirm", "awaiting_post_cancel_choice"
]
sk_count = 0
for k in keys_to_replace:
    text, n1 = re.subn(rf'persisted\.get\("{k}"\)', f'persisted.get(SK.{k})', text)
    text, n2 = re.subn(rf'persisted\["{k}"\]', f'persisted[SK.{k}]', text)
    text, n3 = re.subn(rf'persisted\.pop\("{k}"', f'persisted.pop(SK.{k}', text)
    text, n4 = re.subn(rf'"{k}" in persisted', f'SK.{k} in persisted', text)
    sk_count += (n1+n2+n3+n4)
print(f"Replaced {sk_count} typed keys.")

# Fix LRU cache in agents.py (Phase 4 piece)
text, clru = re.subn(r'@lru_cache(?:.*?)\n\s*def _llm_extract_booking_fields', 'def _llm_extract_booking_fields', text)
print(f"Removed {clru} LRU caches from agents.py")

with open(AGENTS_PATH, 'w', encoding='utf-8') as f:
    f.write(text)


print("\n--- PHASE 4: nlp_extractor.py ---")
if os.path.exists(NLP_PATH):
    with open(NLP_PATH, 'r', encoding='utf-8') as f:
        nlp_text = f.read()

    # Just in case the function is here too
    nlp_text, nn1 = re.subn(r'@lru_cache(?:.*?)\n\s*def _llm_extract_booking_fields', 'def _llm_extract_booking_fields', nlp_text)

    # Inject dynamic_config import if not present
    if 'get_thresholds' not in nlp_text:
        nlp_text = nlp_text.replace('from app.services.config import', 'from app.services.dynamic_config import get_thresholds\nfrom app.services.config import')
        if 'from app.services.dynamic_config import get_thresholds' not in nlp_text:
            nlp_text = 'from app.services.dynamic_config import get_thresholds\n' + nlp_text

    # Replace hardcoded threshold numbers with config values
    # Function 1: _fuzzy_property_type
    # Function 2: _detect_city
    nlp_text, n1 = re.subn(r'0\.94', 'get_thresholds().nlp.fuzzy_match_strict', nlp_text)
    nlp_text, n2 = re.subn(r'0\.90', 'get_thresholds().nlp.fuzzy_match_high', nlp_text)
    nlp_text, n3 = re.subn(r'0\.88', 'get_thresholds().nlp.fuzzy_match_medium', nlp_text)
    nlp_text, n4 = re.subn(r'0\.78', 'get_thresholds().nlp.fuzzy_match_low', nlp_text)
    
    print(f"Replaced {nn1} LRU caches, and {n1+n2+n3+n4} thresholds in nlp_extractor.py")
    with open(NLP_PATH, 'w', encoding='utf-8') as f:
        f.write(nlp_text)


print("\n--- PHASE 5: tracing.py ---")
with open(TRACING_PATH, 'r', encoding='utf-8') as f:
    t_text = f.read()

if 'from opentelemetry import trace' not in t_text:
    t_text = 'from opentelemetry import trace\n' + t_text

import_tracer = "tracer = trace.get_tracer(__name__)\n\n"
if "tracer = trace.get_tracer(__name__)" not in t_text:
    t_text = t_text.replace("class _Span(ContextDecorator):", import_tracer + "class _Span(ContextDecorator):")

# Wipe the old class completely using regex and replace with the proper OTEL implementation
new_span_class = """class _Span(ContextDecorator):
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.span_ctx = None
        self.current_span = None

    def __enter__(self):
        self.span_ctx = tracer.start_as_current_span(self.name)
        self.current_span = self.span_ctx.__enter__()
        for k, v in self.attributes.items():
            self.current_span.set_attribute(k, _safe_attr_value(v))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.current_span.record_exception(exc_value)
        self.span_ctx.__exit__(exc_type, exc_value, traceback)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return self.__exit__(exc_type, exc_value, traceback)"""

# Safe regex replacement
if "class _Span" in t_text:
    # Everything from "class _Span(ContextDecorator):" to "def span(" (not including it)
    pattern = r'class _Span\(ContextDecorator\):(.*?)\ndef span\('
    t_text, span_c = re.subn(pattern, new_span_class + '\n\ndef span(', t_text, flags=re.DOTALL)
    print(f"Replaced tracing.py class: {span_c}")

with open(TRACING_PATH, 'w', encoding='utf-8') as f:
    f.write(t_text)

print("ALL PHASES COMPLETE")

