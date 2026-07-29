"""Tests for stage 5: the reference checker and the stage's own invariants."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from stage5_frontier.check_refs import (
    DEFAULT_GUIDE,
    DEFAULT_ROOT,
    check_refs,
    extract_refs,
    main,
)

STAGE_DIR = Path(__file__).resolve().parents[1]

open_dreamer_missing = not DEFAULT_ROOT.is_dir()
needs_open_dreamer = pytest.mark.skipif(
    open_dreamer_missing, reason=f"open-dreamer checkout not found at {DEFAULT_ROOT}"
)


def test_extract_refs_parses_paths_and_symbols():
    text = (
        "See `dreamer/models.py::KVCache` and `dreamer/utils.py`, plus "
        "`configs/dynamics.yaml`. Ignore `not/a/ref.py` and `KVCache` alone. "
        "Duplicate `dreamer/models.py::KVCache` should collapse."
    )
    refs = extract_refs(text)
    assert refs == [
        ("dreamer/models.py", "KVCache"),
        ("dreamer/utils.py", None),
        ("configs/dynamics.yaml", None),
    ]


@needs_open_dreamer
def test_guide_references_all_resolve():
    refs, errors = check_refs(DEFAULT_GUIDE, DEFAULT_ROOT)
    assert errors == []
    # The tour is only useful if it is substantial.
    assert len(refs) >= 30
    symbol_refs = [r for r in refs if r[1] is not None]
    assert len(symbol_refs) >= 20


@needs_open_dreamer
def test_guide_covers_required_tour_stops():
    required = [
        ("dreamer/utils.py", "TokenLayout"),
        ("dreamer/models.py", "KVCache"),
        ("dreamer/models.py", "BlockCausalTransformer"),
        ("dreamer/models.py", "Tokenizer"),
        ("dreamer/models.py", "Dynamics"),
        ("dreamer/training.py", "shortcut_forcing_step"),
        ("dreamer/generation.py", "DenoiseSchedule"),
        ("dreamer/generation.py", "next_latent"),
        ("dreamer/generation.py", "latent_rollout"),
        ("configs/dynamics.yaml", None),
        ("configs/tokenizer.yaml", None),
    ]
    refs = set(extract_refs(DEFAULT_GUIDE.read_text()))
    missing = [r for r in required if r not in refs]
    assert not missing, f"guide is missing required tour stops: {missing}"


@needs_open_dreamer
def test_checker_fails_on_bogus_path(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("Look at `dreamer/models.py::KVCache` and `dreamer/does_not_exist.py`.")
    _, errors = check_refs(guide, DEFAULT_ROOT)
    assert errors == ["missing file: dreamer/does_not_exist.py"]


@needs_open_dreamer
def test_checker_fails_on_bogus_symbol(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("Look at `dreamer/models.py::TotallyMadeUpClass`.")
    _, errors = check_refs(guide, DEFAULT_ROOT)
    assert errors == ["missing symbol: dreamer/models.py::TotallyMadeUpClass"]


def test_checker_fails_on_missing_root(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("`dreamer/models.py`")
    _, errors = check_refs(guide, tmp_path / "nowhere")
    assert len(errors) == 1 and "root not found" in errors[0]


@needs_open_dreamer
def test_main_exit_codes(tmp_path):
    assert main([]) == 0
    bad = tmp_path / "bad.md"
    bad.write_text("`dreamer/nope.py`")
    assert main(["--guide", str(bad)]) == 1


@needs_open_dreamer
def test_train_smoke_runs():
    result = subprocess.run(
        [sys.executable, "-m", "stage5_frontier.train", "--smoke"],
        capture_output=True,
        text=True,
        cwd=STAGE_DIR.parent,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all references resolve" in result.stdout


def test_all_deliverables_exist():
    for name in [
        "WHY.md",
        "README.md",
        "exercises.md",
        "guide_open_dreamer.md",
        "research.md",
        "wmgym.md",
        "robotics.md",
        "reading.md",
        "check_refs.py",
        "train.py",
    ]:
        assert (STAGE_DIR / name).is_file(), f"missing deliverable: {name}"


def test_no_dash_characters_in_stage_files():
    banned = re.compile("[\u2014\u2013]")
    offenders = []
    for path in STAGE_DIR.rglob("*"):
        if path.suffix in {".md", ".py"} and path.is_file():
            if banned.search(path.read_text()):
                offenders.append(str(path))
    assert not offenders, f"em/en dash found in: {offenders}"
