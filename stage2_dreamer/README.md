# Stage 2: Dreamer (simplified DreamerV3) on CarRacing-v3

A world model with a recurrent state (RSSM: GRU h plus categorical z), a
pixel decoder, reward and continue heads, and an actor-critic trained
entirely inside 15-step imagined rollouts. Backprop through the dream
replaces the evolution of stage 1.

## What's here

    rssm.py     RSSM core: prior, posterior, straight-through categoricals,
                balanced KL with free bits. Hand-write this one first.
    s2_wm.py    Encoder, decoder, reward and continue heads, symlog/symexp,
                combined world model loss.
    s2_ac.py    Actor, critic, imagination rollout, lambda returns, losses.
    buffer.py   Episode replay buffer of uint8 frames, samples length-16
                subsequences.
    train.py    Collect, train world model, train actor-critic, loop.
    WHY.md      The ideas and every design choice.
    exercises.md  Prediction exercises and break-it labs.
    tests/      Fast pytest checks.

## Commands

From the repo root:

    uv run python -m stage2_dreamer.train --smoke   # tiny CPU run, seconds
    uv run pytest stage2_dreamer/tests -q           # all tests, well under 90s

The smoke run collects one seeded 100-step episode (cached under
`data/stage2_dreamer/`), does 3 world model steps and 3 actor-critic steps,
and asserts every loss is finite.

Lab and exercise flags: `--kl-alpha 0.0`, `--det-z`, `--horizon 50`. See
`exercises.md`.

## Full run, one RTX 5090

Defaults in `full_config()` in `train.py`: 500k env steps at action repeat 2,
batch 32 x sequence 16 (pass `--device cuda` for the GPU),
deter 512, depth 32, about 100 world model and 100
actor-critic gradient steps per collected episode, roughly 50k of each over
the run. Expect 6 to 10 hours wall clock. Expected outcome: the car visibly
follows the road; episode returns of 600 to 900 against roughly minus 50 for
random. If returns plateau near zero past 100k steps, check `wm/kl` first
(see the failure modes section of WHY.md).

Modal cost: about 8 hours on an A100 40GB at roughly $2.5/hour, so on the
order of $20 per full run.
