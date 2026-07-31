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
changes. What changes is how well a machine can dream it.

| Stage | Era | What you build |
|:--|:--|:--|
| [0](stage0_compression/) | timeless | Autoencoder, then VAE, on frames |
| [1](stage1_ha_worldmodel/) | 2018 | Ha and Schmidhuber's V + M + C |
| [2](stage2_dreamer/) | 2019-23 | Dreamer: RSSM + actor-critic |
| [3](stage3_token_transformer/) | 2021-24 | VQ tokenizer + transformer (IRIS) |
| [4](stage4_diffusion_forcing/) | 2024-26 | Diffusion forcing + flow matching |
| [5](stage5_frontier/) | now | Frontier lab, research, robotics |

### Why each stage exists

Every arrow below is a specific, reproducible failure. You hit it yourself
before you are shown the fix; that order is the whole method.

**Stage 0: compression is understanding.** A frame is 12k numbers, and the
situation it shows fits in 32.

> A VAE sees one frame at a time. It knows what the world looks like, not
> how it moves. -> **stage 1**

**Stage 1: dreaming is predicting your own latents.** An RNN predicts the
next latent, and a tiny controller trains entirely inside that dream.

> The dream is blurry and short, and the controller learns to cheat it.
> -> **stage 2**

**Stage 2: keep a recurrent state, imagine in latent space.** Backprop the
policy gradient straight through the dream.

> RNN memory becomes the bottleneck, and attention scales better.
> -> **stage 3**

**Stage 3: the world as a language.** Frames become tokens, and dynamics
becomes next-token prediction.

> Autoregressive tokens drift, sampling is slow, and errors compound frame
> by frame. -> **stage 4**

**Stage 4: denoise the future.** Train with noisy context so the model
tolerates its own mistakes.

> No successor yet. This is the frontier. -> **stage 5**

**Stage 5: read the real thing.** Open-dreamer (1.6B params) with stage-4
eyes, plus WMGym, research directions, and the robotics branch.

> Open problems: memory, drift, and agents that cheat their own dreams.

## How to work through it

Each stage directory contains:

- `WHY.md`. The idea in plain language. Every design choice gets its why. Read
  it before the code. It ends with a "you get it when" checklist. Do not move
  on until you can answer that checklist from memory.
- `exercises.md`. Prediction exercises (commit to an answer before running) and
  break-it labs (sabotage the model in a prescribed way, watch it fail, explain
  the failure). The break-it labs are the actual course. The working code is
  just the lab bench.
- Code. Minimal and self-contained, commented where a choice is non-obvious and
  nowhere else.
- `train.py --smoke`. Runs the whole pipeline on CPU in about a minute or two.
  It proves the code is correct, not that the model is good.
- Full-run configs for one RTX 5090, with cost notes for Modal.
- `tests/`. Fast pytest checks. `uv run pytest` from the repo root must stay
  green.

Rules of engagement:

1. Type, don't read. For the core model of each stage (marked in WHY.md), write
   it yourself from the paper description before looking at the reference
   implementation. Struggling first is the mechanism, not an obstacle.
2. Predict before you run. Write down what you expect from every experiment.
   Being wrong is the most valuable signal you can buy with GPU hours.
3. One stage at a time. Stage N assumes stage N-1 is in your bones.
4. Keep a `NOTEBOOK.md` (gitignored, yours) and log everything that confuses
   you. Confusion is the curriculum telling you where to dig.

## Hardware playbook

- Any laptop (CPU is enough): all smoke runs, all tests, all code reading. The
  whole course is developable without a GPU.
- One modern GPU (16GB or more, a 4090/5090 class card is comfortable): every
  full training run in stages 0-4 fits on a single card, from about 30 minutes
  (stage 0) to about 8-11 hours (stage 4).
- A multi-GPU box, if you have access to one: stage 4/5 data-parallel runs and
  sweeps. This is where the open-dreamer blog's MFU and roofline lessons become
  real. Entirely optional.
- Cloud (Modal, or any provider with A100/H100 rentals): burst capacity if you
  have no local GPU. Each stage's docs say what a full run costs. Doing every
  full run on rented A100s lands somewhere around $60-120 total.

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
