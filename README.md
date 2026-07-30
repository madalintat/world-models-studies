# world-models-studies

One environment. Five generations of world models. Build every one of them.

The point is not to run code. The point is to end up with the whole field in
your head: why latents, why imagination, why tokens, why diffusion, and why
each idea replaced the one before it. Every stage here exists because the
previous stage fails in a specific, visible way. You hit the failure yourself
first, then you learn the fix. That order is the method.

## The thread

The environment is CarRacing-v3 (a car on a procedurally generated track;
frames come out at 96x96 and every stage resizes them to 64x64). It never
changes. What changes is how well a machine can
dream it.

| Stage | Idea from | What you build | The one idea | The failure that motivates the next stage |
|---|---|---|---|---|
| 0 | timeless | Autoencoder, then VAE, on single frames | Compression is understanding. A frame is 12k numbers; the situation it shows fits in 32. | A VAE sees one frame at a time. It knows what the world looks like, not how it moves. |
| 1 | 2018 | Ha and Schmidhuber's V + M + C | Dreaming is predicting your own latents. An RNN predicts the next latent, and a tiny controller trains entirely inside that dream. | The dream is blurry and short, and the controller learns to cheat it. |
| 2 | 2019-2023 | Dreamer (RSSM plus actor-critic in imagination) | Keep a recurrent state, imagine in latent space, backprop through the dream. | RNN memory becomes the bottleneck, and attention scales better. |
| 3 | 2021-2024 | VQ tokenizer plus transformer (IRIS style) | The world as a language: frames become tokens, dynamics becomes next-token prediction. | Autoregressive tokens drift, sampling is slow, errors compound frame by frame. |
| 4 | 2024-2026 | Diffusion forcing with flow matching (mini open-dreamer) | Denoise the future, and train with noisy context so the model tolerates its own mistakes. | None yet. This is the frontier. |
| 5 | now | Frontier lab | Read the real open-dreamer (1.6B params) with stage-4 eyes. WMGym, research directions, robotics branch. | Open problems: memory, drift, agents cheating dreams. |

## How to work through it

Each stage directory contains:

- `WHY.md`. The idea in plain language. Every design choice gets its why.
  Read it before the code. It ends with a "you get it when" checklist. Do not
  move on until you can answer that checklist from memory.
- `exercises.md`. Prediction exercises (commit to an answer before running)
  and break-it labs (sabotage the model in a prescribed way, watch it fail,
  explain the failure). The break-it labs are the actual course. The working
  code is just the lab bench.
- Code. Minimal and self-contained, commented where a choice is non-obvious
  and nowhere else.
- `train.py --smoke`. Runs the whole pipeline on CPU in about a minute or
  two. It proves the code is correct, not that the model is good.
- Full-run configs for one RTX 5090, with cost notes for Modal.
- `tests/`. Fast pytest checks. `uv run pytest` from the repo root must stay
  green.

Rules of engagement:

1. Type, don't read. For the core model of each stage (marked in WHY.md),
   write it yourself from the paper description before looking at the
   reference implementation. Struggling first is the mechanism, not an
   obstacle.
2. Predict before you run. Write down what you expect from every experiment.
   Being wrong is the most valuable signal you can buy with GPU hours.
3. One stage at a time. Stage N assumes stage N-1 is in your bones.
4. Keep a `NOTEBOOK.md` (gitignored, yours) and log everything that confuses
   you. Confusion is the curriculum telling you where to dig.

## Hardware playbook

- Any laptop (CPU is enough): all smoke runs, all tests, all code reading.
  The whole course is developable without a GPU.
- One modern GPU (16GB or more, a 4090/5090 class card is comfortable):
  every full training run in stages 0-4 fits on a single card, from about
  30 minutes (stage 0) to about 8-11 hours (stage 4).
- A multi-GPU box, if you have access to one: stage 4/5 data-parallel runs
  and sweeps. This is where the open-dreamer blog's MFU and roofline lessons
  become real. Entirely optional.
- Cloud (Modal, or any provider with A100/H100 rentals): burst capacity if
  you have no local GPU. Each stage's docs say what a full run costs. Doing
  every full run on rented A100s lands somewhere around $60-120 total.

## Setup

```bash
cd world-models-studies
uv sync          # CPU env, everything smoke-testable
uv run pytest    # should be green before you start
```

For a CUDA machine, see `SETUP.md` for the GPU environment swap.

## Where this goes

After stage 5 there are three live exits, all mapped in `stage5_frontier/`:

- Research: drift, memory, and evaluation ablations that fit on 8x5090.
- The repo next door: `../open-dreamer`. After stage 4 you can read all 6.3k
  lines of its `dreamer` package, and its missing agent-training loop is a
  known, wanted, unclaimed contribution.
- Robotics: the same ideas on MuJoCo and Meta-World, with a WMGym leaderboard
  submission as a concrete milestone.
