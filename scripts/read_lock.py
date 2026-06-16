#!/usr/bin/env python3
"""Print a field from external_revisions.lock (single source of truth for pins).

Used by setup scripts to default to the locked upstream commit:

    python scripts/read_lock.py dino_wm commit   # -> <sha>
    python scripts/read_lock.py r2dreamer repo   # -> https://github.com/NM512/r2dreamer.git

Exits non-zero (and prints nothing to stdout) if the section/key is absent, so a
shell caller can fall back cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

LOCK_PATH = Path(__file__).resolve().parents[1] / "external_revisions.lock"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: read_lock.py <section> [key=commit]", file=sys.stderr)
        return 2
    section = argv[1]
    key = argv[2] if len(argv) > 2 else "commit"
    try:
        data = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"cannot read {LOCK_PATH}: {exc}", file=sys.stderr)
        return 1
    entry = data.get(section)
    if not isinstance(entry, dict) or key not in entry:
        print(f"no {section}.{key} in {LOCK_PATH.name}", file=sys.stderr)
        return 1
    print(entry[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
