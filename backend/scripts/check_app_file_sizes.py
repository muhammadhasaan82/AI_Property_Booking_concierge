"""Fail if any Python file under app/ exceeds the line budget (excluding test/eval paths)."""
from __future__ import annotations

import sys
from pathlib import Path

MAX_LINES = 900
ROOT = Path(__file__).resolve().parents[1] / "app"
SKIP_PARTS = {
    "tests",
    "evals",
    "evaluation",
    "__pycache__",
    "migrations",
}
SKIP_SUFFIXES = {".pyc"}


def should_check(path: Path) -> bool:
    if path.suffix != ".py" or path.name.endswith(tuple(SKIP_SUFFIXES)):
        return False
    return not any(part in SKIP_PARTS for part in path.parts)


def main() -> int:
    offenders: list[tuple[int, Path]] = []
    for path in sorted(ROOT.rglob("*.py")):
        if not should_check(path):
            continue
        count = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
        if count > MAX_LINES:
            offenders.append((count, path))

    if not offenders:
        print(f"OK: no Python files under {ROOT} exceed {MAX_LINES} lines.")
        return 0

    print(f"FAIL: {len(offenders)} file(s) exceed {MAX_LINES} lines:\n")
    for count, path in sorted(offenders, reverse=True):
        rel = path.relative_to(ROOT.parent)
        print(f"  {count:5d}  {rel}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
