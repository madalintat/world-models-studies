"""Repo-wide markdown formatting rules.

The course docs are meant to be read in a terminal, a narrow editor pane, and
on GitHub without horizontal scrolling. Two rules keep that true:

  - prose wraps at 79 columns
  - fenced or indented code wraps at 90 (commands have atomic tokens that
    cannot always be broken, so they get more slack than sentences)

Tables are exempt: GFM has no line continuation inside a cell, so an overlong
table row is a signal to restructure the table, not to wrap it. The test
still reports them, capped at a width where a row stops fitting on screen.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PROSE_LIMIT = 79
CODE_LIMIT = 90
TABLE_LIMIT = 100

FENCE = re.compile(r"^\s*(```|~~~)")
TABLE = re.compile(r"^\s*\|")

SKIP_DIRS = {"node_modules", "runs", "data", "__pycache__"}


def is_ours(path: Path) -> bool:
    """Course docs only: no dot-directories (.venv, .git, .pytest_cache) and
    no generated output trees."""
    parts = path.relative_to(REPO_ROOT).parts
    return not any(p.startswith(".") or p in SKIP_DIRS for p in parts)


def markdown_files() -> list[Path]:
    return sorted(p for p in REPO_ROOT.rglob("*.md") if is_ours(p))


def classify(lines: list[str]):
    """Yield (line_number, text, kind) with kind in {prose, code, table}."""
    in_fence, fence_token = False, None
    for number, line in enumerate(lines, start=1):
        fence = FENCE.match(line)
        if fence:
            if in_fence and fence.group(1) == fence_token:
                in_fence = False
            elif not in_fence:
                in_fence, fence_token = True, fence.group(1)
            yield number, line, "code"
            continue
        if in_fence or line.startswith("    ") or line.startswith("\t"):
            yield number, line, "code"
        elif TABLE.match(line):
            yield number, line, "table"
        else:
            yield number, line, "prose"


LIMITS = {"prose": PROSE_LIMIT, "code": CODE_LIMIT, "table": TABLE_LIMIT}


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.name))
def test_line_widths(path: Path):
    rel = path.relative_to(REPO_ROOT)
    offenders = [
        f"{rel}:{number} ({kind}, {len(line)} > {LIMITS[kind]}): {line[:60]}..."
        for number, line, kind in classify(path.read_text().split("\n"))
        if len(line) > LIMITS[kind]
    ]
    assert not offenders, "overlong lines:\n" + "\n".join(offenders)


def test_docs_are_ascii():
    """The course docs are plain ASCII, matching the stage-file dash rule."""
    offenders = []
    for path in markdown_files():
        for number, line in enumerate(path.read_text().split("\n"), start=1):
            non_ascii = [c for c in line if ord(c) > 127]
            if non_ascii:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{number} {sorted(set(non_ascii))}")
    assert not offenders, "non-ASCII characters:\n" + "\n".join(offenders)
