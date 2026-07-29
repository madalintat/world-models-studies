"""Verify that every open-dreamer reference in guide_open_dreamer.md is real.

The guide references files in ../open-dreamer with backticked paths like
`dreamer/models.py` or `dreamer/models.py::KVCache`. This script extracts
every such reference, checks that the file exists under the open-dreamer
checkout, and, when a ::Symbol suffix is present, checks that the file
actually defines that symbol as a class or function. Exits nonzero on any
missing path or symbol, so the guide cannot silently rot as the upstream
repo evolves.

Usage:
    uv run python -m stage5_frontier.check_refs
    uv run python -m stage5_frontier.check_refs --guide path/to/guide.md --root path/to/open-dreamer
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
DEFAULT_GUIDE = STAGE_DIR / "guide_open_dreamer.md"
DEFAULT_ROOT = STAGE_DIR.parent.parent / "open-dreamer"

# Backticked paths under the three top-level source dirs, with an optional
# ::Symbol anchor. Anything else in backticks (shell commands, config keys,
# symbols without a path) is deliberately ignored.
REF_PATTERN = re.compile(
    r"`((?:dreamer|configs|scripts)/[A-Za-z0-9_./-]+\.(?:py|yaml|md))"
    r"(?:::([A-Za-z_][A-Za-z0-9_]*))?`"
)


def extract_refs(guide_text: str) -> list[tuple[str, str | None]]:
    """Return (path, symbol_or_None) pairs in order of appearance, deduplicated."""
    seen: set[tuple[str, str | None]] = set()
    refs: list[tuple[str, str | None]] = []
    for match in REF_PATTERN.finditer(guide_text):
        ref = (match.group(1), match.group(2))
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def check_refs(guide_path: Path, root: Path) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Check every reference. Returns (all_refs, error_messages)."""
    errors: list[str] = []
    if not guide_path.is_file():
        return [], [f"guide not found: {guide_path}"]
    if not root.is_dir():
        return [], [f"open-dreamer root not found: {root}"]

    refs = extract_refs(guide_path.read_text())
    if not refs:
        errors.append(f"no references found in {guide_path}; the parser or the guide is broken")

    for rel_path, symbol in refs:
        target = root / rel_path
        if not target.is_file():
            errors.append(f"missing file: {rel_path}")
            continue
        if symbol is not None:
            text = target.read_text(errors="replace")
            if not re.search(rf"(?:class|def)\s+{re.escape(symbol)}\b", text):
                errors.append(f"missing symbol: {rel_path}::{symbol}")
    return refs, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    refs, errors = check_refs(args.guide, args.root)
    print(f"checked {len(refs)} references against {args.root}")
    if errors:
        for err in errors:
            print(f"  BROKEN {err}")
        return 1
    print("all references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
