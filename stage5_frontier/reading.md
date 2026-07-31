# Reading list, in course order

One honest sentence each on why the paper earns its slot. Read them after
the stage that makes them legible, not before; papers are compressed and
the stages are the decompressor.

## Stage 1's roots

- Ha and Schmidhuber, "World Models" (2018). The paper that named the
  field and the source of stage 1 wholesale: VAE plus MDN-RNN plus a tiny
  controller trained inside the dream, still the clearest statement of the
  core bet, and the interactive site (worldmodels.github.io) is half the
  value.

## Stage 2's line

- Hafner et al., "Learning Latent Dynamics for Planning from Pixels"
  (PlaNet, 2019). Introduces the RSSM you built in stage 2; read it for
  the deterministic-plus-stochastic state argument, and do not skip the
  planning half. Dreamer replaced CEM with a learned actor and the field
  followed, but planning on top of a learned model came back: Simchowitz
  lab's nano-world-model (2026) ships CEM model-predictive control over a
  diffusion world model as one of three headline applications, and it
  trains no policy at all. Sampling action sequences and scoring them with
  the model is the cheapest way to turn any of your stages 1 through 4
  into a controller.
- Hafner et al., "Dream to Control" (DreamerV1, 2020). Replaces PlaNet's
  planner with an actor-critic trained by backprop through imagined
  latent trajectories, which is the single move that made world models a
  training method rather than a curiosity.
- Hafner et al., "Mastering Diverse Domains through World Models"
  (DreamerV3, 2023). The robustness paper: symlog, two-hot critics, KL
  balancing, free bits, one hyperparameter set across 150 tasks, and the
  honest lesson is that most of the gap between a paper idea and a
  working system is this kind of unglamorous normalization engineering.

## Stage 3's line

- van den Oord et al., "Neural Discrete Representation Learning"
  (VQ-VAE, 2017). Where the codebook and the straight-through estimator
  you fought in stage 3 come from; read it to see how little of the
  original framing was about world models at all.
- Micheli et al., "Transformers are Sample-Efficient World Models"
  (IRIS, 2023). Stage 3's blueprint: frames to VQ tokens, dynamics as
  next-token prediction with a transformer, and strong Atari-100k results
  that made the tokens-plus-transformer recipe respectable.

## Stage 4's line

- Chen et al., "Diffusion Forcing" (2024). The pivotal reframe of the
  course: give every frame its own noise level, unifying teacher forcing
  and full-sequence diffusion, and making noisy-context training the
  drift defense you measured.
- Lipman et al., "Flow Matching for Generative Modeling" (2022). The
  clean mathematical core under stage 4: regress a velocity field along
  straight noise-to-data paths, no ELBOs, no score matching scaffolding,
  and diffusion becomes three readable equations.
- Frans et al., "One Step Diffusion via Shortcut Models" (2024). Where
  step-size conditioning and the two-half-steps-teach-one-full-step
  bootstrap loss come from; open-dreamer's shortcut_forcing_step is this
  paper fused with diffusion forcing.

## The frontier (stage 5's subjects)

- Hafner, Yan, Lillicrap, "Training Agents Inside of Scalable World
  Models" (Dreamer 4, 2025). The paper open-dreamer implements: shortcut
  forcing, a causal tokenizer, and an agent trained inside the world
  model at Minecraft scale, worth reading with the code open in a second
  window.
- "Genie 3" (Google DeepMind, 2025). The promptable-world direction:
  real-time interactive generated environments at 720p with
  minutes-scale persistence, thin on method details, but it defines what
  "world models as products" currently means and what the memory probe
  in research.md is poking at.
- Assran et al., "V-JEPA 2" (2025). The other church: predict in
  representation space, never reconstruct pixels, then do zero-shot
  robot manipulation on top; read it as the strongest current argument
  that stage 0's "reconstruction equals understanding" premise is
  optional.
- The open-dreamer blog post (next-state.github.io/open-dreamer). The
  training log for the codebase you toured in this stage, and the rare
  document that reports MFU, rooflines, data plumbing, and failure
  modes at a scale a small team can actually replicate; read it last,
  after the code, and it reads like a colleague's postmortem instead of
  an announcement.
- Huang et al., "Nano World Models: A Minimalist Implementation of
  Future Video Prediction" (2026, arXiv 2605.23993). The paper for the
  repo toured in `guide_nano_world_model.md`, and the most useful thing
  on this list per page: it ablates prediction target, action injection,
  and model scale head to head and publishes the tables, so it functions
  as an answer key to stage 4's prediction exercises. Read it for the
  method of the ablations rather than the winners.

## Keeping this list alive

This list is deliberately short and stops where the course stops. Two
public indexes track the field week by week and are the right place to
go when you want breadth instead of sequence:

- github.com/nik-55/world-models. A curated table of reports, code and
  blogs, currently reaching into 2026, plus links onward to the larger
  Awesome-World-Model lists. Useful precisely because it is dated: you
  can see how fast the interactive-world-generation line is moving
  against the robotics line.
- The benchmark entries there are the part to watch, because the honest
  bottleneck in this field is evaluation, not architecture. WorldScore,
  World-in-World, Omni-WorldBench and WM-Gym (see `wmgym.md`) are all
  attempts at the same problem: scoring a world model by whether its
  predictions are good enough to act on, rather than by how the frames
  look.

A warning about lists like these. They are optimized for recall, not
precision, and reading them front to back is a good way to feel busy
while learning nothing. Use one when you have a specific question
(who else conditions on actions this way, what is the state of long-
horizon memory) and ignore it otherwise. The stages are the
decompressor; a list of titles is not.
