import re

with open(r'c:\Users\ASUS\Desktop\Hotel booking\services\agents.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Phase 1: Eradicate Receipt Monolith
print("Phase 1: Replacing monolithic receipt blocks...")
import_line = 'from services.confirmation_helpers import _render_receipt\n'
if import_line not in text:
    text = import_line + text

# Use regex to find receipt blocks
# They start with `receipt = f"""📋 **BOOKING SUMMARY**` or similar and end with `"""`
# Also there are non-fstring ones like `receipt=f"""📋 **BOOKING SUMMARY**`
# We'll use a more flexible regex:
receipt_pattern = r'receipt\s*=\s*f\"\"\"📋 \*\*BOOKING SUMMARY\*\*(?:.|\n)*?\"\"\"'
matches = re.findall(receipt_pattern, text)
print(f"Found {len(matches)} receipt blocks to replace.")
text = re.sub(receipt_pattern, 'receipt = _render_receipt(persisted)', text)

# Phase 2: Eliminate Hard-coded Config Arrays
print("Phase 2: Replacing hard-coded required field arrays...")
import_config_line = 'import services.config as config\n'
if import_config_line not in text:
    if 'import services.config' not in text:
        text = import_config_line + text

# Replace inline arrays
# required=["name", "phone", "email", "check_in", "check_out", "guests"]
# Can have different spacing.
arr1_pattern = r'\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*\]'
arr1_matches = re.findall(arr1_pattern, text)
print(f"Found {len(arr1_matches)} inline required arrays (without selected_property).")
text = re.sub(arr1_pattern, 'config.REQUIRED_FIELDS', text)

# required=["name","phone","email","check_in","check_out","guests",SK.selected_property]
arr2_pattern = r'\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*,\s*SK\.selected_property\s*\]'
arr2_matches = re.findall(arr2_pattern, text)
print(f"Found {len(arr2_matches)} inline required arrays (WITH selected_property).")
text = re.sub(arr2_pattern, 'config.REQUIRED_FIELDS + [SK.selected_property]', text)

# Phase 3: Enforce Typed State Keys
print("Phase 3: Enforcing typed state keys...")
# The user specifically mentioned:
# persisted.get("awaiting_post_mod_choice") -> persisted.get(SK.awaiting_post_mod_choice)
# persisted["receipt_shown"] = True -> persisted[SK.receipt_shown] = True
# persisted["awaiting_field"] = "modification_choice" -> persisted[SK.awaiting_field] = "modification_choice"
# "modifying_dates" -> SK.modifying_dates

replacements = [
    (r'persisted\.get\("awaiting_post_mod_choice"\)', r'persisted.get(SK.awaiting_post_mod_choice)'),
    (r'persisted\["receipt_shown"\]', r'persisted[SK.receipt_shown]'),
    (r'persisted\["awaiting_field"\]', r'persisted[SK.awaiting_field]'),
    (r'persisted\.get\("awaiting_field"\)', r'persisted.get(SK.awaiting_field)'),
    (r'persisted\.pop\("awaiting_field", None\)', r'persisted.pop(SK.awaiting_field, None)'),
    (r'"modifying_dates"', r'SK.modifying_dates'),
    (r'persisted\.get\("modifying_dates"\)', r'persisted.get(SK.modifying_dates)'),
    (r'persisted\["modifying_dates"\]', r'persisted[SK.modifying_dates]'),
    (r'persisted\.pop\("modifying_dates", None\)', r'persisted.pop(SK.modifying_dates, None)'),
    (r'"receipt_shown"', r'SK.receipt_shown'),
    (r'"awaiting_post_mod_choice"', r'SK.awaiting_post_mod_choice'),
    (r'"awaiting_field"', r'SK.awaiting_field'),
]

for old, new in replacements:
    before = len(text)
    text = re.sub(old, new, text)

# Write back
with open(r'c:\Users\ASUS\Desktop\Hotel booking\services\agents.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Phases 1, 2, 3 complete for agents.py.")
