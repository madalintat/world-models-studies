# Stage 4: Diffusion forcing with flow matching in latent space

A mini open-dreamer: a small conv AE compresses 64x64 CarRacing frames to an
8x8x8 latent grid (64 tokens of dim 8), and a factorized space-time
transformer learns latent dynamics with flow matching, where every frame in
a training sequence gets its own independent noise level. Rollouts use a
K-step tau ladder with re-noised context and are scored with the same
PSNR-vs-horizon drift curve as stage 3.

Read `WHY.md` first. Write `flow.py` yourself before opening the provided
one. Then do `exercises.md`.

## Files

- `flow.py`: flow matching core (interpolation, per-frame tau sampling,
  weighted x-space loss, tau ladder, Euler mixing, context noising) plus
  the inference-time scheduling matrix. One function in here,
  `_pyramid_schedule`, is deliberately left for you to write; break-it lab
  D in `exercises.md` has the brief and three tests skip until you do.
- `s4_latent_ae.py`: deterministic conv AE, mirrors stage 0's VAE minus the
  variational part
- `s4_model.py`: temporal transformer, per-frame tokens
  [action, tau, 64 latents], alternating full-within-frame and
  causal-across-time attention, with the action-injection mechanism
  selectable (token, additive, FiLM)
- `sampling4.py`: autoregressive tau-ladder rollout (`rollout`) and
  scheduled block rollout (`rollout_block`), drift curve, gif
- `train.py`: end to end (collect, AE, dynamics, rollout, drift csv)
- `tests/test_stage4_diffusion_forcing.py`

## Commands

    # everything tiny, CPU, under two minutes
    uv run python -m stage4_diffusion_forcing.train --smoke

    # variants used by the exercises
    uv run python -m stage4_diffusion_forcing.train --smoke --clean-context
    uv run python -m stage4_diffusion_forcing.train --smoke --weighting v_space
    uv run python -m stage4_diffusion_forcing.train --smoke --tau-ctx 0.5
    uv run python -m stage4_diffusion_forcing.train --smoke --k-steps 1

    # how the action reaches the model (nano-world-model's ablation axis)
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --injection additive
    uv run python -m stage4_diffusion_forcing.train --smoke --injection film

    # inference-time scheduling, and the SD3 noise-level distribution
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --schedule full_sequence
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --schedule pyramid --stagger 1
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --tau-sampling logit_normal

    # tests
    uv run pytest stage4_diffusion_forcing/tests -q

Outputs land in `data/stage4_diffusion_forcing/out_smoke/`: `drift.csv`
(PSNR per generated frame), `rollout.gif` (ground truth left, rollout
right), `checkpoint.pt`. Collected frames are cached in
`data/stage4_diffusion_forcing/` and reused across runs; delete the npz to
recollect.

## Full run on one RTX 5090

Drop `--smoke` and pass `--device cuda` to get the full config (documented
defaults in `train.py`):
100k frames at seed 0, AE batch 256 for 30k steps (about 45 minutes), then
the dynamics model (d_model 512, depth 12, 8 heads, sequences of 32 frames,
batch 32) for 250k steps with v_space weighting recommended. Expect 8 to 11
hours total. Outcome to expect: 4-step rollouts hold above roughly 20 dB
PSNR past frame 50 with stable road geometry; compare against your stage 3
curve at the same horizon, which decayed steadily.

Modal cost: about 10 hours on one H100 at roughly $4/h, on the order of $40
(an L40S halves the price for a slower run).

Scaling to 8 GPUs: plain DDP data parallel, one replica per GPU with batch
32 each and gradient all-reduce. No model parallelism is worth it at this
size; the same holds for open-dreamer, whose training is data parallel for
the same reason (the dynamics model is small, the batch is what you scale).
