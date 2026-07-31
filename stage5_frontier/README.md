# Stage 5: the frontier lab

No training code in this stage. The deliverables are documents grounded in
the real open-dreamer checkout at `../open-dreamer`, plus one script that
keeps them honest.

## What's here

- `WHY.md`: why this stage exists, failure modes, the "you get it when"
  checklist. Read first. It also names the file to write yourself before
  reading mine.
- `guide_open_dreamer.md`: a guided reading tour of `../open-dreamer`,
  mapping every major file and class to the course stage that taught it.
  Every file and symbol reference is machine-verified.
- `guide_nano_world_model.md`: a shorter tour of the Simchowitz lab's
  nano-world-model, the closest published sibling to your stage 4. Read it
  for its ablation tables, which are answer keys to stage 4's prediction
  exercises, and for the two ideas the course does not have: semantic
  latent codecs with no decoder, and CEM planning on top of a world model.
  References are to the public repo and are not machine-checked.
- `research.md`: three experiment specs sized for an 8-GPU workstation
  (drift benchmark, memory probe, dream-cheating study), with metrics,
  GPU-hour budgets, and what results would mean.
- `wmgym.md`: participation plan for the WM-Gym benchmark
  (https://wm-gym.labs.reka.ai/).
- `robotics.md`: the MuJoCo branch: dm_control, Meta-World, MJX/ManiSkill,
  and how stage 2's code ports.
- `reading.md`: the papers, in course order, one honest sentence each.
- `exercises.md`: five prediction exercises and three break-it labs, all
  runnable on this laptop.
- `check_refs.py`: parses `guide_open_dreamer.md` for open-dreamer file and
  symbol references and exits nonzero if any do not exist in the checkout.
- `tests/test_stage5.py`: pytest coverage for the checker and the stage's
  own style rules.

## Commands

```bash
cd world-models-studies   # the repo root

# verify every reference in the guide against ../open-dreamer
uv run python -m stage5_frontier.check_refs

# course-uniform smoke entry point (prints the stage map, runs the checker)
uv run python -m stage5_frontier.train --smoke

# tests (a few seconds, CPU)
uv run pytest stage5_frontier/tests -q
```

Requires an open-dreamer checkout next to this repo (clone
github.com/next-state/open-dreamer as `../open-dreamer`); `--root` on the
checker overrides the location, and the tests skip cleanly if it is absent.

## Full-run guide

There is nothing to train in this stage, so "full run" means the work the
documents define:

- One RTX 5090: not the target for this stage; a single card covers the
  smoke/eval slices of the research.md experiments (probe-set
  construction, rollout evaluation, one arm of the dream-cheating ladder
  at about 2-4 GPU-hours per world model) while the multi-run comparisons
  want the full box.
- 8x RTX 5090: the three research.md experiments total roughly 270
  GPU-hours, about 2.5 days of the full box. Per-experiment batch sizes,
  step counts, wall clocks, and expected outcomes are documented in
  `research.md` and summarized in `train.py`'s docstring.

Modal cost note: the full research program is about $300-500 on rented
A100-40GB; the reading tour, exercises, tests, and smoke run cost nothing
and run on this laptop.
