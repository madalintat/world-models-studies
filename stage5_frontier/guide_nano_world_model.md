# A guided reading of nano-world-model

`guide_open_dreamer.md` tours the biggest thing you can read after stage 4.
This tours the closest thing. nano-world-model is the Simchowitz lab's
minimalist diffusion-forcing video world model (arXiv 2605.23993, MIT
licensed, checkpoints on Hugging Face). It is roughly 26k lines of Python,
but the part that matters to you is small: a DiT-style denoiser, a flow
matching module, and a scheduling utility. You have written all three.

Its value to this course is not architectural. It is evidential. Stage 4
asks you to predict things and then measure them at smoke scale on one
environment, where most effects are inside the noise. nano-world-model ran
the same questions properly, at scale, across six datasets, and published
the tables. It is the answer key to exercises you have already attempted.

Conventions: references below are paths inside the public repository, so
unlike the open-dreamer guide they are not machine-checked by
`check_refs.py`; clone it yourself if you want to follow along.

    git clone https://github.com/simchowitzlabpublic/nano-world-model

Two differences from the course worth naming up front:

1. Their latents come from a pretrained Stable Diffusion VAE or from a
   frozen semantic encoder (DINO, V-JEPA 2), never from an autoencoder they
   trained themselves. Stage 0 through 4 always trained the compressor.
   That difference is the subject of stop 4.
2. They target robot and game datasets (DINO-WM, RT-1, CSGO), not
   CarRacing. Where you have one environment and a drift curve, they have
   six and a metric suite.

## Stop 1: `src/diffusion/flow_matching.py`

Read this first because you will recognize it line for line. It is your
`flow.py`, and the differences are all conventions rather than ideas.

Their forward process is `x_t = tau * x_0 + (1 - tau) * eps` with
`tau = 1 - (t + 1) / T`, so tau = 1 is data and tau = 0 is noise: exactly
your convention. Their training target is the constant vector field
`u = x_0 - eps`, where yours is `x_0` itself. That is the x-prediction
versus v-prediction distinction from your WHY.md, and their
`_predict_xstart` is the algebra converting between them:
`x_0 = x_t + (1 - tau) * u`. Their Euler step is
`x_next = x + (tau_next - tau_curr) * u`; your `euler_mix` is the same
step written in terms of the x-prediction, which is why your `make_ladder`
betas exist at all.

Two things to actually look at:

- The comment near the end of `dfot_ddim_sample` documenting a bug they
  fixed: the final Euler step has to send tau to 1.0 for a frame marked
  clean, and an earlier version sent it to 0.0, which jumped the latent
  from nearly-clean back to pure noise and silently undid the whole
  rollout. Your `make_ladder` avoids this by construction, because its last
  beta is exactly 0 so the final latent is the model's last x-prediction.
  Go and confirm that for yourself, then notice that a convention bug of
  this kind produces plausible-looking garbage rather than a crash.
- `training_losses` returns unweighted MSE, with a comment that flow
  matching needs no SNR weighting. Hold that against your `loss_weight`
  and its `v_space` scheme. Both positions are defensible; the point is
  that your weighting exists because you predict x, and theirs is absent
  because they predict u. Weighting and prediction target are the same
  decision wearing two hats.

## Stop 2: `src/diffusion/df_sample.py`

This is the file stage 4 was missing until you wrote `scheduling_matrix`,
and it is the best single argument for reading this repo.

`generate_full_sequence_schedule`, `generate_pyramid_schedule` and
`generate_sequential_schedule` are your three modes, and their worked
examples in the docstrings are the same shape as the one in your
`_pyramid_schedule`. Then `ddim_idx_to_timestep` maps ladder indices onto
real diffusion timesteps with -1 meaning clean, which is the bookkeeping
your tau ladder gets for free by working in continuous tau.

Three things they have that you do not, in increasing order of interest:

- `n_generate_frames` truncates the schedule as soon as the frames you
  actually intend to keep are clean. In a sliding-window rollout where you
  keep only the first generated frame per step, everything after that row
  is wasted compute.
- `history_stabilization_level` re-noises the context frames to a fixed
  level and pins them there for the whole sampling loop. This is your
  `tau_ctx` and `noise_context`, arrived at independently, which is a
  useful thing to notice about how load-bearing that idea is.
- A comment in `generate_scheduling_matrix` recording that they used to
  generate the schedule for the full horizon and mask context frames to -1
  afterwards, wasting `n_context_frames * sampling_timesteps` rows in
  sequential mode. Your `rollout_block` prepends nothing and budgets rows
  only for generated frames. Read their fix as a warning about what a
  schedule bug costs: not correctness, just silent slowness.

## Stop 3: `src/models/nanowm.py` and the injection ablation

Their denoiser is a DiT: patchify, RoPE, multi-head attention with QK
RMSNorm, SwiGLU feed-forwards, adaLN-conditioned blocks, and alternating
spatial and temporal attention. The spatial/temporal alternation is your
`AxialBlock`, and the factorization argument is identical. Note that
`Attention` here is plain multi-head, not the grouped-query attention
open-dreamer uses; at 40M to 460M parameters the KV cache is not yet the
thing that hurts.

Go to `TransformerBlock.forward` and read the five action-injection
branches: `additive`, `adaln`, `adaln_fuse`, `film`, `cross_attention`.
Your `s4_model.py` now implements three of these, and their `film`
modulation head is zero-initialized exactly like yours, for the same
reason. Then read the tables in `docs/training.md#design-choices`, which
give the head-to-head numbers on RT-1 and PushT.

The number worth carrying: on PushT's 2-dimensional actions, all five
mechanisms land within 0.32 PSNR of each other, and plain additive wins
with zero extra parameters. On RT-1's 7-dimensional end-effector actions
FiLM pulls ahead on FID. Cross-attention is worst in both, despite costing
the most parameters. CarRacing's action is three numbers, so stage 4's
exercise P5 is being run in the regime where the axis is nearly dead, and
your measured result should say so.

That is the transferable skill: an ablation table is evidence about a
regime, not a ranking of mechanisms. Their own default (`additive`) is a
statement about their datasets.

## Stop 4: `src/latent_codecs/`

Here is the genuinely new idea, and it is a challenge to stage 0's premise.

`sd_vae.py` wraps a pretrained Stable Diffusion VAE: a reconstruction
codec, the same kind of object as your stage 0 VAE and stage 4 conv AE,
only someone else trained it on far more data. But `semantic.py` defines
`DensePatchLatentCodec` with `has_decoder = False`, wrapping DINO or
V-JEPA 2 features. Read that class attribute and sit with it. The world
model predicts future latents in a space from which no image can be
recovered at all.

Stage 0's WHY.md argues that compression is understanding, and it proves
it by reconstruction: the bottleneck must retain what matters because the
decoder has to rebuild the frame. A codec with no decoder abandons that
argument entirely and keeps only the prediction task. Your stage 5
`reading.md` calls V-JEPA 2 "the other church"; this is that church with a
working implementation you can read in 223 lines.

The honest question to leave with, because nobody has settled it: if you
cannot decode, how do you know your latents did not throw away the thing
you will need in fifty frames? Their answer is downstream task
performance, which is also WM-Gym's answer, which is also `wmgym.md`'s
argument for why decision regret beats pixel metrics.

## Stop 5: `src/planning/cem_planner.py`

This is the stop that changed a line in this course's own reading list.

`CEMPlanner.plan` samples action sequences from a Gaussian, rolls each one
forward through the world model, scores the resulting trajectories against
an objective, keeps the elites, refits the Gaussian, and repeats. That is
the cross-entropy method, it is a few hundred lines, and it turns a pure
video predictor into a controller with no policy training whatsoever.

Stage 2 taught you the other answer: train an actor and a critic inside the
dream and backpropagate through it. Dreamer's move, and the reason the
field mostly stopped planning. But planning has real advantages that stage
2's approach does not: nothing to train, the objective can change at test
time, and it works on any model that can roll forward, including your
stages 1, 3 and 4, none of which have a policy.

The cost is inference-time compute, `num_samples` rollouts per decision per
CEM iteration, which is exactly why the scheduling matrix of stop 2 stops
being an academic concern. Planning is where cheap sampling becomes the
whole ballgame.

If you want a concrete next project after this stage, this is the cheapest
good one: put a CEM planner on top of your stage 4 model, use the drift
curve horizon you already trust as the planning horizon, and measure real
CarRacing return against stage 2's actor-critic at equal wall clock.

## Stop 6: `src/utils/metrics.py` and `docs/evaluation.md`

Your drift curve is PSNR against a ground-truth continuation. That is one
number, and it has a known blind spot: PSNR rewards blur, because the
blur-minimizing prediction is the conditional mean and MSE loves the mean.
Stage 0's WHY.md taught you exactly this about reconstruction, and then
stage 3 and 4 went ahead and scored rollouts with PSNR anyway.

They report four metrics, and the split is instructive: PSNR and SSIM
compare your frame to the true frame; LPIPS compares deep features, so it
notices when a frame is sharp but wrong in a way PSNR forgives; FID and FVD
compare your *distribution* of frames or clips to the real one, and need no
paired ground truth at all. Their headline table shows Rope and Granular
with respectable PSNR and much worse FID, which is the signature of a model
producing individually-plausible frames that collectively miss the data
distribution. No pairwise metric can see that.

Cheapest upgrade to this course: LPIPS alongside PSNR in the stage 3 and
stage 4 drift curves. It is one dependency and it would make `research.md`'s
drift experiment considerably harder to fool.

## What to take away

1. Your stage 4 has no errors that this repo reveals. The flow matching
   algebra, the ladder, and the context re-noising all match. That is worth
   knowing, and it is not what you should expect from a from-scratch build.
2. What it has instead are omissions, and they cluster at inference:
   scheduling, cheap sampling, and evaluation.
3. The ablation tables are the reusable artifact. Read them as answer keys,
   and read the defaults as claims about a regime.

When their `docs/training.md` tables read as confirmations of things you
already measured, rather than as new information, this stop is done.
