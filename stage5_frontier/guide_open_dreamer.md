# A guided reading of open-dreamer

You finished stages 0 through 4. This document walks you through the real
thing: `../open-dreamer`, a JAX/Flax NNX implementation of the Dreamer 4
world-model pipeline, trained on Minecraft/VPT gameplay and playable in real
time. Roughly 8k lines of Python, 6.3k of them in the `dreamer` package. There
is nothing in it you have not already built in miniature. The point of this
tour is to prove that to you, file by file.

Conventions: every reference below is written as `dir/file.py` or
`dir/file.py::Symbol`. Run `uv run python -m stage5_frontier.check_refs` and it
will confirm that every one of them exists in the checkout right now. No line
numbers anywhere, on purpose: line numbers rot, symbols mostly don't.

Two things are genuinely different from the course and worth naming up front:

1. It is JAX, not torch. Read `nnx.Module` as `nn.Module`, `jax.lax.scan` as a
   compiled for-loop, and pytree registration as "this object can pass through
   jit". None of the ideas change.
2. The environment is Minecraft at 360x640, not CarRacing at 64x64. The
   pipeline shape (tokenize, then model dynamics in latent space, then roll
   out) is exactly your stage 3 and stage 4 shape.

Read in the order below. Budget a few evenings.

## Stop 1: `dreamer/utils.py::TokenLayout`

Start here because every model file leans on it.

Stage 3 taught you that a frame becomes a sequence of tokens and that position
in the sequence carries meaning. `TokenLayout` is that idea grown up: a single
timestep is an ordered list of (modality, count) segments, and
`dreamer/utils.py::Modality` names the kinds: LATENT, IMAGE, ACTION, PROPRIO,
REGISTER, SPATIAL, SHORTCUT, AGENT. What is new at scale is that the layout
also owns the attention rules: `dreamer/utils.py::build_space_mask` builds a
per-frame mask from the modality ids, with three modes. In "encoder" mode,
latent tokens may attend to everything while patch tokens only see their own
kind (information is funneled into the latents). In "decoder" mode it is
reversed: latents are read-only sources and patch queries pull from them. In
"wm_agent" mode there is a strict hierarchy: actions see only actions,
observations see observations and actions, agent tokens see all. That last mask
is the architectural slot where an actor-critic will live; hold that thought
until stop 8.

While you are in the file: `dreamer/utils.py::patchify` and
`dreamer/utils.py::unpatchify` are the stage 0 conv encoder replaced by "cut
the image into 16x16 squares and call each one a token", and
`dreamer/utils.py::normalize_latents` plus
`dreamer/utils.py::pack_bottleneck_to_spatial` are the unglamorous bookkeeping
that stage 4 taught you matters (your latents had to be normalized before
noising too). `dreamer/utils.py::build_ema_model` and
`dreamer/utils.py::ema_update_step` maintain the EMA copy of the dynamics
model; stage 4's bootstrap targets needed the same trick for stability.

## Stop 2: `dreamer/models.py::KVCache`

Stage 3 concept: autoregressive sampling is unbearable unless you cache keys
and values instead of re-encoding the whole past every step. What is new at
scale: this cache is a ring buffer with a statically known shape so it can live
inside jit. `dreamer/models.py::KVCache` writes with a fast contiguous path and
a slow wrapping path (see the `update` method), and reads through
`get_ordered_kv`, which rolls the buffer so the newest entry is last and then
builds two masks: a validity mask so attention cannot read the zero-filled
slots during warmup, and a shifted causal mask. When your stage 3 sampler just
sliced a Python list, this is what that convenience costs once you need a
fixed-size sliding window under a compiler.
`dreamer/models.py::create_transformer_caches` allocates one cache per
time-attention layer (space layers need none; you will see why at stop 3).

## Stop 3: `dreamer/models.py::BlockCausalTransformer`

Stage 3 concept: a causal transformer predicts the future from the past. What
is new at scale: attention is factorized. A
`dreamer/models.py::BlockCausalLayer` is either a space layer
(`dreamer/models.py::SpaceSelfAttention`: full attention among the tokens of
one frame, time folded into the batch) or a time layer
(`dreamer/models.py::TimeSelfAttention`: causal attention across frames, one
spatial position at a time, space folded into the batch). The stack interleaves
them, one time layer every `time_every` layers; with the dynamics config (depth
30, time_every 4, offset 0) that is 8 time layers and 22 space layers. This is
why full frame-by-frame attention over 290 tokens times 192 frames never has to
be materialized. The building blocks are current-generation language-model
parts you have seen in the wild: `dreamer/models.py::GroupedQueryAttention`
(fewer KV heads than query heads), `dreamer/models.py::RotaryEmbedding1D`
(RoPE), RMSNorm, and gradient checkpointing via `nnx.remat`. Compare with your
stage 3 transformer: same causal masking idea, plus an engineering
decomposition that your 64x64 world never forced on you. Also note
`estimate_attention_flops` at the bottom: at this scale you budget FLOPs before
you launch, which is stage 5's whole attitude.

## Stop 4: `dreamer/models.py::Tokenizer`

Stage 0 concept: compression is understanding; a frame's content fits in a
small latent. Stage 3 refined it: tokenize, then model dynamics on tokens. What
is new at scale, in three moves:

- The latent is continuous, not a VQ codebook. `dreamer/models.py::Encoder`
  appends 512 learned latent tokens to the 920 patch tokens of each frame, runs
  the block-causal transformer with the "encoder" mask from stop 1, and
  squeezes the latent tokens through a tanh bottleneck of width 16
  (`d_bottleneck`). Your stage 3 VQ-VAE quantized; this just bounds. The
  discrete codebook, and all its collapse failure modes from your stage 3
  break-it lab, is simply gone, because the dynamics model downstream (stage 4
  style, diffusion) does not need discrete targets. Only a stage 3 next-token
  model does.
- The encoder is regularized by masking: `dreamer/models.py::MAEReplacer`
  randomly replaces up to 90 percent of patch tokens with a learned mask token
  during training, so reconstruction cannot rely on copying pixels through.
  This plays the role that the KL term played in your stage 0 VAE: it is the
  thing that stops the autoencoder from cheating.
- It tokenizes video, not images: the time layers are causal, so latents for
  frame t may use frames before t. Your stage 0 model saw one frame at a time
  and the WHY.md there told you what it therefore could not know; this encoder
  is allowed to know it.

`dreamer/models.py::Decoder` mirrors it with learned per-patch query tokens and
the "decoder" mask, and `dreamer/models.py::Tokenizer` glues encode and decode
together. Training lives in `scripts/train_tokenizer.py` with an MSE plus LPIPS
objective (see `lpips_weight` in `configs/tokenizer.yaml`); LPIPS is the
perceptual patch on the blur problem you first saw in stage 0's reconstructions
and stage 1's dreams.

## Stop 5: `dreamer/models.py::Dynamics`

Stage 4 concept: the world model is a denoiser. Given noisy latents, the
action, and the noise level, predict the clean latents (x-prediction). What is
new at scale, walking the `__call__`:

- Latent tokens arrive unpacked as 512 tokens of width 16 and are packed two at
  a time (`packing_factor`) into 256 spatial tokens, then projected to d_model
  1920. Fewer, fatter tokens: pure compute economics.
- `dreamer/models.py::ActionEncoder` embeds a structured action
  (`dreamer/actions.py::Actions` holds binary keys, a categorical mouse bucket,
  and continuous values) into a single token. Your stage 4 model concatenated a
  3-vector for steer, gas, brake; Minecraft's action space needs a real
  encoder, and `dreamer/actions.py::shift_actions` prepends a no-op so action t
  aligns with the frame it produces.
- A single shortcut token carries the noise conditioning:
  `dreamer/models.py::TimestepEmbedder` sinusoidally embeds the step index
  (which rung of the ladder) and the signal level tau, concatenated. Stage 4
  conditioned on a per-frame noise level; this adds the step size because of
  shortcut training (stop 6).
- 32 learned register tokens per timestep give attention scratch space, and the
  "wm_agent" mask from stop 1 wires everything together, including the
  still-empty AGENT slot.
- The output head `flow_x_head` is zero-initialized, so at step 0 the model
  predicts "clean latent equals zero" in normalized space, a calm start you
  also used (or should have) in stage 4.

Time attention runs with a sliding window of `context_length` 192 frames
(`configs/dynamics.yaml`). That window is the model's entire memory; the memory
probe in `research.md` exists because of this line.

## Stop 6: `dreamer/training.py::shortcut_forcing_step`

This is the heart of the repo, and it is your stage 4 training step with one
addition. Read the helpers first:

- `dreamer/training.py::sample_tau_for_step` samples per-frame signal levels
  tau on a discrete grid; independent noise per frame is exactly diffusion
  forcing as you built it, and tau equal to 1 (clean) is included so the model
  also learns from teacher-forced frames.
- `dreamer/training.py::compute_flow_loss` is x-space MSE between predicted and
  true clean latents, weighted by `dreamer/training.py::loss_weight` (the
  default `v_space` scheme reweights x-space MSE so it behaves like a
  velocity-space loss; you derived the x-pred versus v-pred tradeoff in stage
  4's break-it lab).
- `dreamer/training.py::apply_ot_coupling` pairs the sampled Gaussian noise
  with data by a Sinkhorn optimal-transport match inside the minibatch, so flow
  paths cross less. A refinement your stage 4 skipped; straighter paths matter
  more when you want very few sampling steps.

Now the addition. The batch is split into empirical rows and bootstrap rows
(`B_self`). Empirical rows train plain diffusion forcing at the finest step
size. Bootstrap rows, sampled at coarser step sizes via
`dreamer/training.py::sample_step_excluding_dmin`, train the shortcut
objective: take two half-steps with a frozen (EMA) copy of the model, and teach
the student to reach the same endpoint in one full step
(`dreamer/training.py::compute_bootstrap_loss`, with the algebra rearranged
into x-space to dodge a divide-by-almost-zero). This is self-distillation
toward few-step sampling, and it is why the demo can run at 4 denoising steps
per frame in real time. Per `configs/dynamics.yaml`, bootstrap rows switch on
only after step 100k of 200k (`bootstrap_start`): first learn the flow, then
learn to skip along it.

`dreamer/training.py::compute_psnr` and `dreamer/training.py::run_evaluation`
are your stage 3 drift measurement industrialized: periodic teacher-forced and
free-running rollouts, PSNR per horizon, saved side-by-side videos. The outer
loop in `scripts/train_dynamics.py` is recognizable from every stage you built:
iterate batches, jit the step, EMA update, checkpoint, evaluate.

## Stop 7: `dreamer/generation.py`

Sampling. Three functions, in the order you should read them:

`dreamer/generation.py::DenoiseSchedule` precomputes the tau ladder: for k
sampling steps, signal levels tau from 0 to 1 and mixing coefficients beta
where beta equals (1 - tau_next) / (1 - tau_current). Convince yourself with
stage 4 eyes that the update "x becomes beta x plus (1 - beta) x0_pred" is an
Euler step on the straight-line flow, and that the betas telescope so the
product of betas up to step s equals 1 - tau_s (exercise 3 makes you do this).

`dreamer/generation.py::next_latent` denoises one future frame: start from pure
noise, run the ladder with `jax.lax.scan`, each rung calling the dynamics model
with the KV cache from stop 2. Then comes the single most instructive detail in
the repo: the finalized latent is NOT written to the cache clean. It is
re-noised to tau_ctx (about 0.9, snapped onto a valid ladder rung by
`DenoiseSchedule.init`) and pushed through the model once more to produce the
cache entry. Train with noisy context, then serve noisy context: the model
never sees a cleaner past at inference than it saw in training, and its own
small errors on past frames look like noise it was explicitly trained to
tolerate. This is the anti-drift mechanism of stage 4, stated in two lines of
sampling code.

`dreamer/generation.py::latent_rollout` is the free-running dream: prefill the
cache with ground-truth context at tau 1 (clean, finest step index), then scan
`next_latent` for the horizon, feeding actions.
`dreamer/generation.py::next_frame` is the interactive variant (one action in,
one decoded frame out, caches carried), and `dreamer/sampler.py::sample_video`
is the end-to-end convenience: frames to latents to rollout to decoded video,
ground-truth reconstruction alongside for honest comparison. Evaluation against
real videos is FVD, in `dreamer/fvd/fvd.py`, driven by `scripts/eval_fvd.py`.

## Stop 8: configs, and the seam where stage 2 plugs in

Read `configs/tokenizer.yaml` and `configs/dynamics.yaml` last, top to bottom;
every field is commented, and after stops 1 through 7 each comment reads as a
decision you have already faced. Highlights worth pausing on:

- Scale: dynamics depth 30 with d_model 1920 (config arithmetic ties width to
  depth) works out to roughly 1.1B parameters, and the tokenizer checkpoint the
  dynamics config points at is named tokenizer-500M, so the full pipeline sits
  around 1.6B. Typed dataclasses backing the YAML live in
  `dreamer/configs.py::DynamicsModelConfig` and
  `dreamer/configs.py::TokenizerModelConfig`.
- Optimization: Muon for matrix weights with AdamW for the rest, a
  warmup-stable-decay schedule, EMA decay 0.999. Compare with the Adam defaults
  you used all course; at this scale optimizer choice is a measured, budgeted
  decision, and `dreamer/scaling.py::compute_max_steps` plus
  `dreamer/scaling.py::ScalingContext` exist to spend FLOPs deliberately
  (Chinchilla-style token-per-parameter budgeting).
- Data: `dreamer/data/data.py::build_iterator` and
  `dreamer/data/data.py::PackEpisodes` pack short clips into long training
  sequences with block-causal masks so no batch slot is wasted; format details
  in `dreamer/data/README.md`. `configs/dataset/minecraft_vpt.yaml` and
  `configs/dataset/minecraft_vpt_latent.yaml` are the raw and tokenized dataset
  descriptions; `scripts/tokenize_minecraft_dataset.py` converts one into the
  other offline, exactly your stage 3 two-phase pipeline.
- Infrastructure you never needed on one card:
  `dreamer/parallel.py::build_parallel` and `dreamer/parallel.py::MeshRules`
  describe data/FSDP/tensor sharding, and
  `dreamer/checkpointing.py::CheckpointBundle` wraps Orbax checkpoints.
  `configs/common.yaml` holds shared precision and logging defaults (bfloat16
  compute, float32 params), with logging in `dreamer/logging.py`.

And the seam. The README's roadmap lists one unchecked box: the Dreamer 4
behaviour-cloning / RL agent training loop. Now look at what is already plumbed
and waiting: `Modality.AGENT` in `dreamer/utils.py`, the "wm_agent" mask
hierarchy where agent tokens read everything, the `task_embeddings` argument
threaded through `dreamer/models.py::Dynamics` and
`dreamer/training.py::shortcut_forcing_step`, the agent hidden states
`h_states` already returned from the training step, and the `p_include_reward`
knob in `dreamer/configs.py`. There is no policy head, no value head, no
imagination-rollout objective. That missing loop is stage 2 of this course,
verbatim: an actor and critic trained inside the frozen world model's dream.
You built it against an RSSM; here it would train against stops 5 through 7. It
is a known, wanted, unclaimed contribution, and after this tour you know
exactly where every wire goes.

## The map, compressed

Each entry is one piece of open-dreamer, the course stage that taught it,
and what upstream added on top of what you built.

- `dreamer/utils.py::TokenLayout`: **stage 3.**
  Frames as ordered tokens, now multi-modality with mask rules.
- `dreamer/models.py::KVCache`: **stage 3.**
  Sampling cache, now a jit-safe ring buffer with a sliding window.
- `dreamer/models.py::BlockCausalTransformer`: **stage 3.**
  Causal transformer, now factorized into space and time layers.
- `dreamer/models.py::Tokenizer`: **stages 0 and 3.**
  Compression, now a continuous tanh bottleneck plus MAE masking, no VQ.
- `dreamer/models.py::Dynamics`: **stage 4.**
  Action-conditioned denoiser, now with shortcut and register tokens.
- `dreamer/training.py::shortcut_forcing_step`: **stage 4.**
  Diffusion forcing plus shortcut self-distillation for 4-step sampling.
- `dreamer/generation.py::next_latent`: **stage 4.**
  Tau ladder plus re-noised context: drift tolerance served at inference.
- `dreamer/training.py::run_evaluation`: **stage 3.**
  Your drift PSNR measurement, industrialized.
- The missing agent loop: **stage 2.**
  Actor-critic in imagination, and the open contribution.
- `configs/dynamics.yaml`: **all stages.**
  Every course decision, written down as a commented field.

When the map above feels obvious rather than clever, you are done with the
course and ready for `research.md`.
