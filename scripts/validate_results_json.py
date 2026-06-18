"""Validate a results.json file against the FTR output schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.output.schema import validate_output_schema  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_results_json.py <path/to/results.json>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 1

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON: {exc}", file=sys.stderr)
        return 1

    errors = validate_output_schema(data)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"\n{len(errors)} error(s). FAILED.", file=sys.stderr)
        return 1

    print(f"OK: {path} is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
