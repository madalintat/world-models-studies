"""Stage 5 has no training. This entry point exists so the course-wide
smoke harness stays uniform: it runs the reference checker over
guide_open_dreamer.md and prints the stage map.

--smoke and the default mode do the same thing, in about a second, on CPU.

FULL "run" for this stage (documentation only; nothing here executes it):
the GPU work of stage 5 is the three experiments in research.md, sized for
the 8x RTX 5090 box. Reference numbers, per experiment:

  1. drift benchmark    stage 3 vs stage 4 retrains at matched compute,
                        batch 64 sequences of 32 frames, 60k steps each,
                        3 seeds per model, about 130 GPU-hours total,
                        expected wall clock about 17 hours on 8 cards.
                        Expected outcome: stage 4's PSNR-vs-horizon curve
                        decays visibly slower than stage 3's.
  2. memory probe       one stage 4 retrain per context length in
                        {8, 32, 128}, batch 32, 40k steps, about 60
                        GPU-hours, under a day on 8 cards. Expected
                        outcome: re-entry IoU stays near the permutation
                        baseline until context covers the off-screen gap.
  3. dream cheating     12 stage 2 world-model checkpoints of varying
                        quality plus actor training in each, about 80
                        GPU-hours, about 10 hours on 8 cards. Expected
                        outcome: the imagined-vs-real return gap shrinks
                        with model quality but does not reach zero.

Modal cost: all three experiments, about 270 GPU-hours, land near $300-500
on rented A100-40GB at roughly $1.10-1.90/h; the smoke run costs nothing.
"""

from __future__ import annotations

import argparse
import sys

from stage5_frontier.check_refs import DEFAULT_GUIDE, DEFAULT_ROOT, main as check_main

STAGE_FILES = [
    ("WHY.md", "why this stage exists and what it changes about how you work"),
    ("guide_open_dreamer.md", "guided reading tour of ../open-dreamer"),
    ("research.md", "three experiments sized for 8x RTX 5090"),
    ("wmgym.md", "participation plan for the WM-Gym benchmark"),
    ("robotics.md", "the MuJoCo branch: dm_control, Meta-World, MJX/ManiSkill"),
    ("reading.md", "papers in course order"),
    ("exercises.md", "prediction exercises and break-it labs"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5 smoke entry point.")
    parser.add_argument("--smoke", action="store_true",
                        help="accepted for course-wide uniformity; same as the default")
    parser.parse_args(argv)

    print("stage 5: the frontier lab. No training here; the deliverables are documents")
    print("grounded in the open-dreamer checkout, and the experiments in research.md.")
    print()
    for name, desc in STAGE_FILES:
        print(f"  {name:24s} {desc}")
    print()
    print(f"verifying {DEFAULT_GUIDE.name} against {DEFAULT_ROOT} ...")
    return check_main([])


if __name__ == "__main__":
    sys.exit(main())
