"""Quick smoke test for TOON module and NLP engine imports."""
import sys, os
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === Test TOON ===
from services.toon import toon_encode, toon_decode

# 1. Simple object
obj1 = {"name": "John", "age": 30, "active": True}
t1 = toon_encode(obj1)
d1 = toon_decode(t1)
assert d1["name"] == "John"
assert d1["age"] == 30
assert d1["active"] is True
print("PASS: simple object round-trip")

# 2. Colon in value
obj2 = {"time": "12:30:00"}
t2 = toon_encode(obj2)
d2 = toon_decode(t2)
assert d2["time"] == "12:30:00"
print("PASS: string with colons")

# 3. Newline in value
obj3 = {"msg": "line1\nline2"}
t3 = toon_encode(obj3)
assert "\\n" in t3
d3 = toon_decode(t3)
assert d3["msg"] == "line1\nline2"
print("PASS: string with newlines")

# 4. Null, bool, empty
obj4 = {"a": None, "b": True, "c": False, "d": []}
t4 = toon_encode(obj4)
d4 = toon_decode(t4)
assert d4["a"] is None
assert d4["b"] is True
assert d4["c"] is False
assert d4["d"] == []
print("PASS: null, bool, empty array")

# 5. Nested object
obj5 = {"user": {"name": "Jane", "city": "NYC"}}
t5 = toon_encode(obj5)
d5 = toon_decode(t5)
assert d5["user"]["name"] == "Jane"
print("PASS: nested object")

# 6. Numeric string preserved
obj6 = {"phone": "12345"}
t6 = toon_encode(obj6)
d6 = toon_decode(t6)
assert d6["phone"] == "12345"
assert isinstance(d6["phone"], str)
print("PASS: numeric string preserved")

# 7. Key with colon
obj7 = {"time:stamp": "2024"}
t7 = toon_encode(obj7)
d7 = toon_decode(t7)
assert d7["time:stamp"] == "2024"
print("PASS: key with colon")

# === Test NLP Engine imports ===
from services import nlp_engine

# Affirmation
assert nlp_engine.classify_affirmation("yes") == "yes"
assert nlp_engine.classify_affirmation("no") == "no"
assert nlp_engine.classify_affirmation("whatever") == "neutral"
print("PASS: classify_affirmation")

assert nlp_engine.is_greeting("hi") is True
assert nlp_engine.is_greeting("apartment") is False
print("PASS: is_greeting")

assert nlp_engine.is_acknowledgment("ok") is True
print("PASS: is_acknowledgment")

assert nlp_engine.is_end_request("bye") is True
assert nlp_engine.is_end_request("hello") is False
print("PASS: is_end_request")

assert nlp_engine.extract_email("my email is john@example.com") == "john@example.com"
print("PASS: extract_email")

dates = nlp_engine.extract_dates("2027-06-01 to 2027-06-05")
assert "2027-06-01" in dates
print("PASS: extract_dates")

assert nlp_engine.extract_cardinal("3") == 3
assert nlp_engine.extract_cardinal("first") == 1
print("PASS: extract_cardinal")

print()
print("=" * 50)
print("ALL SMOKE TESTS PASSED")
print("=" * 50)
