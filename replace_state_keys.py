"""
Automated find-and-replace: raw state key strings → SK.xxx references.
Run from the project root.
"""
import re, os

ROOT = r"c:\Users\ASUS\Desktop\Hotel booking\services"

# State key mappings: raw string → SK attribute
KEYS = {
    '"awaiting_field"': "SK.awaiting_field",
    '"awaiting_selection_confirm"': "SK.awaiting_selection_confirm",
    '"awaiting_post_mod_choice"': "SK.awaiting_post_mod_choice",
    '"awaiting_post_cancel_choice"': "SK.awaiting_post_cancel_choice",
    '"receipt_shown"': "SK.receipt_shown",
    '"modifying_dates"': "SK.modifying_dates",
    '"selected_property"': "SK.selected_property",
    '"recent_selection_index"': "SK.recent_selection_index",
    '"recent_property_id"': "SK.recent_property_id",
}

FILES = ["agents.py", "graph.py", "confirmation_helpers.py"]

IMPORT_LINE = "from .state_keys import SK"

for fname in FILES:
    path = os.path.join(ROOT, fname)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add import if not present
    if IMPORT_LINE not in content:
        # Insert after existing imports
        # Find the last "from ." import line
        lines = content.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from .") or stripped.startswith("from services."):
                insert_idx = i + 1
            elif stripped.startswith("import ") and i < 20:
                insert_idx = i + 1
        lines.insert(insert_idx, IMPORT_LINE)
        content = "\n".join(lines)
    
    # Replace all state key strings
    count = 0
    for raw, typed in KEYS.items():
        # Don't replace inside comments or docstrings that explain the key
        # Also don't replace in string formatting like f"...{key}..."
        # Simple approach: replace all occurrences
        old_content = content
        content = content.replace(raw, typed)
        changes = content.count(typed) - old_content.count(typed)
        if changes > 0:
            count += changes
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"{fname}: {count} replacements")
