import re

with open(r'c:\Users\ASUS\Desktop\Hotel booking\services\agents.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure we have the imports
import_helper = "from services.confirmation_helpers import _render_receipt\n"
if "from services.confirmation_helpers import _render_receipt" not in content:
    content = import_helper + content
    
if "import services.config as config" not in content:
    content = "import services.config as config\n" + content

# Phase 1: The giant receipt blocks.
# Let's see the structure. It usually starts with `            receipt=f"""📋 **BOOKING SUMMARY**` or similar.
# There is also `            receipt = f"""📋 **BOOKING SUMMARY**`
# We'll use a regex that matches from `receipt[\s=]+f\"\"\"📋` up to the next `\"\"\"`

old_len = len(content)
pattern1 = r'receipt\s*=\s*f\"\"\"📋(?:.*?)(?<!\\)\"\"\"'
content, count1 = re.subn(pattern1, 'receipt = _render_receipt(persisted)', content, flags=re.DOTALL)
print(f"Replaced {count1} instance(s) of receipt f-strings.")

# Phase 2: Required Fields
pattern2 = r'\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*\]'
content, count2 = re.subn(pattern2, 'config.REQUIRED_FIELDS', content)
print(f"Replaced {count2} instance(s) of REQUIRED_FIELDS.")

pattern3 = r'\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*,\s*SK\.selected_property\s*\]'
content, count3 = re.subn(pattern3, 'config.REQUIRED_FIELDS + [SK.selected_property]', content)
print(f"Replaced {count3} instance(s) of REQUIRED_FIELDS + SK.")

# Note: Sometimes it's `"selected_property"` in the raw array before Phase 3
pattern4 = r'\[\s*"name"\s*,\s*"phone"\s*,\s*"email"\s*,\s*"check_in"\s*,\s*"check_out"\s*,\s*"guests"\s*,\s*"selected_property"\s*\]'
content, count4 = re.subn(pattern4, 'config.REQUIRED_FIELDS + [SK.selected_property]', content)
print(f"Replaced {count4} instance(s) of REQUIRED_FIELDS + 'selected_property'.")

# Phase 3: Typed State Keys replacement in agents.py
print("Doing state keys replacements...")
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
    (r'"selected_property"', r'SK.selected_property'),
    (r'"recent_property_id"', r'SK.recent_property_id'),
    (r'"recent_selection_index"', r'SK.recent_selection_index'),
    (r'"awaiting_selection_confirm"', r'SK.awaiting_selection_confirm'),
    (r'"awaiting_post_cancel_choice"', r'SK.awaiting_post_cancel_choice'),
]

sk_count = 0
for old, new in replacements:
    # Use re.sub to avoid modifying string literals that might be in log messages, 
    # but the simplest way is standard replaced for dictionary keys in bracket notation or get()
    pass

# We can safely use the generic replace script from before for this part, or just simple string replace:
for old, new in replacements:
    content, c = re.subn(old, new, content)
    sk_count += c

# Also general string replacements for dictionary access:
for key in ["awaiting_field", "awaiting_selection_confirm", "awaiting_post_mod_choice",
            "awaiting_post_cancel_choice", "receipt_shown", "modifying_dates", 
            "selected_property", "recent_selection_index", "recent_property_id"]:
    
    # persisted.get("X")
    content, c1 = re.subn(rf'persisted\.get\("{key}"\)', f'persisted.get(SK.{key})', content)
    # persisted["X"]
    content, c2 = re.subn(rf'persisted\["{key}"\]', f'persisted[SK.{key}]', content)
    # persisted.pop("X"
    content, c3 = re.subn(rf'persisted\.pop\("{key}"', f'persisted.pop(SK.{key}', content)
    
    sk_count += c1 + c2 + c3

print(f"Replaced {sk_count} state key accesses.")

with open(r'c:\Users\ASUS\Desktop\Hotel booking\services\agents.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done writing agents.py")
