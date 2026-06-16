#!/usr/bin/env python3
"""Fail if release-facing docs/report text contain real unfinished markers.

This replaces a naive ``grep "placeholder"`` that false-positived on ordinary
prose ("...env placeholders resolved...", "may contain `TBD`", "episode_XXX.npy").
It flags genuine unfinished-work markers only, and ignores anything inside an
inline code span (`like this`) or a fenced code block (```...```), since those
intentionally show config/placeholder syntax or filename patterns.

Scope: the documentation/report/config surface intended for handoff. Source code
under src/ and scripts/ is not scanned (normal code TODOs are not release
blockers, and this script itself names the markers).

    python scripts/check_release_markers.py

Exit 0 = clean; 1 = at least one real marker found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Whole-word markers that signal unfinished work. (Intentionally no "XXX": it
# collides with filename patterns like episode_XXX.npy.)
MARKERS = [r"\bTODO\b", r"\bFIXME\b", r"\bTBD\b", r"\bPLACEHOLDER_FIGURE\b",
           r"\bPLACEHOLDER_TABLE\b", r"<INSERT"]
MARKER_RE = re.compile("|".join(MARKERS))
INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Release-facing files. src/ and scripts/ (code) are deliberately excluded.
INCLUDE_GLOBS = [
    "README.md",
    "REPRODUCIBILITY.md",
    "external_revisions.lock",
    "reports/*.tex",
    "reports/*.md",
    "reports/*.json",
    "reports/*.csv",
    "docs/*.md",
    "configs/**/*.yaml",
]
EXCLUDE_DIR = ROOT / "reports" / "drafts"


def iter_files() -> list[Path]:
    seen: list[Path] = []
    for pattern in INCLUDE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and EXCLUDE_DIR not in path.parents:
                seen.append(path)
    return seen


def scan(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    in_fence = False
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = INLINE_CODE_RE.sub("", raw)  # drop inline `code`
        match = MARKER_RE.search(line)
        if match:
            hits.append((lineno, match.group(0)))
    return hits


def main() -> int:
    found = False
    for path in iter_files():
        for lineno, marker in scan(path):
            found = True
            print(f"{path.relative_to(ROOT)}:{lineno}: unfinished marker {marker!r}")
    if found:
        print("\nFAIL: unfinished markers in release files (see above).", file=sys.stderr)
        return 1
    print("OK: no unfinished markers in release files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
