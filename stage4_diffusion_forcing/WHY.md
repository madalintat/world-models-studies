# Stage 4: Diffusion forcing with flow matching, a mini open-dreamer

## The core idea

Stage 3's transformer was trained to predict the next frame from a perfect
history and then asked, at rollout time, to predict from its own imperfect
history. That mismatch is why it drifts. This stage removes the mismatch: we
train a denoiser that never gets to assume its context is perfect, because
during training every frame in the sequence is independently corrupted by a
random amount of noise. A model trained this way has learned, from day one,
to generate the next frame from context that is partly wrong. When you then
roll it out on its own generations, nothing is out of distribution.

That is the whole stage. Everything else (flow matching, the tau ladder, the
weighting schemes) is machinery to make that idea trainable and cheap to
sample from.

Write `flow.py` yourself before reading mine. It is about 80 lines and it is
the file where every idea in this stage lives; if you can write it from the
descriptions below, you understand the stage.

## Flow matching in plain words

Pick a data point x and a sample of pure Gaussian noise. Draw a straight
line between them: z_tau = (1 - tau) * noise + tau * x. At tau = 0 you are
at the noise end, at tau = 1 you are at the data. Training is: pick a random
tau, hand the model the point z_tau on the line plus the value of tau, and
ask it to name the clean endpoint x. That is it. No score functions, no SDE,
no noise schedules with alphas and bars. A straight line and a regression
target.

We use x-prediction: the model outputs its best guess of the clean x
directly. This is the easiest parametrization to reason about, because the
output is always an image-like latent you can decode and look at, at any
noise level, after any number of steps. The alternatives (predicting the
noise, or predicting the velocity x - noise) are linear reparametrizations
of the same thing, but their outputs are not directly interpretable and
noise-prediction in particular becomes ill-conditioned near tau = 1.

Sampling inverts the line in K steps. Start from pure noise at tau = 0.
At each rung of the ladder, ask the model for x_hat, then move to the next
rung by re-interpolating: keep the fraction beta_s = (1 - tau_{s+1}) /
(1 - tau_s) of your current z and mix in (1 - beta_s) of x_hat. Algebra
(do it once by hand): this is exactly "solve for the implied noise, then
re-draw the line to x_hat at the next tau". At the last rung beta is 0, so
the sample is exactly the model's final x-prediction. With K = 1 the whole
ladder collapses to a single forward pass, which is one of the tests.

## Per-frame independent noise is the whole trick

In stage 3, training always presented clean context. In this stage the
training step samples an independent tau, uniform on [0, 1], for every frame
of every sequence. Frame 3 might be nearly clean at tau 0.95 while frame 4
is nearly pure noise at tau 0.1, and the model must still predict every
frame's clean latent from what it can see.

Look at what this means for frame t under the causal attention mask: its
context, frames 0 through t-1, is a set of latents corrupted by random,
unknown-in-advance amounts. That is a simulation of inference, where the
context is your own generations and is wrong by some amount you cannot
know. A model trained this way stops trusting context blindly. It learns how
much to rely on each context frame from that frame's tau token, and it
learns to fall back on the action sequence and on generic dynamics when
context is unreliable. This is what kills stage 3's compounding drift, and
it costs nothing: same architecture, same data, one changed line in how tau
is sampled. The break-it lab where you set context tau to 1 during training
and watch the rollout fall apart is the single most instructive experiment
in this course.

## x-prediction vs v-prediction, and the v_space weighting

For the linear path the velocity is v = x - noise, and given z_tau the two
errors are related by (v_hat - v) = (x_hat - x) / (1 - tau). So a plain MSE
in v-space equals the x-space MSE times 1 / (1 - tau)^2. That factor is the
entire difference between the two parametrizations' training signals.

Plain x-space MSE underweights high tau: when the input is almost clean,
copying the input already gives a small x-error, so the model gets little
pressure to fix the fine details that the last ladder steps depend on.
v-prediction fixes the weighting but its outputs are not directly usable
and it can be less stable early in training. The v_space weighting scheme,
1 / max(1 - tau, eps)^2 applied to the x-space MSE, recovers v-prediction's
training signal exactly while keeping x-prediction's stable, interpretable
output. Same trick as in the open-dreamer code (`loss_weight` in
`dreamer/training.py`). The ramp scheme, 0.1 + 0.9 * tau, is a gentler
version of the same idea. `--weighting` switches between all three.

## Why noise the context at inference too

At rollout, previously generated frames go back into the context re-noised
to a fixed tau_ctx (0.9 by default), and their tau tokens say tau_ctx, not
1. Two reasons. First, distribution matching: training never promised the
model clean context, so handing it generated frames labeled as clean (tau 1)
is a lie twice over: the frames are not clean, and tau = 1 context is rare
under uniform sampling. Second, the added noise actively washes out the
small errors in generated frames before they can be interpreted as signal.
The model was trained to treat a tau 0.9 frame as 90 percent trustworthy,
which is about what a generated frame deserves. Ground-truth prefill frames
stay clean at tau 1, matching open-dreamer's `next_latent`, which noises
only the frames the model decoded itself.

## Why few sampling steps matter, and what comes next

Each ladder step is a full forward pass, per frame. A world model you can
drive interactively needs each frame in tens of milliseconds, so K = 4 vs
K = 64 is the difference between playable and slideshow. Diffusion forcing
already gets useful samples at K = 4 here. The next step, which this stage
describes but does not implement, is shortcut distillation: train the model
to also take coarser rungs directly, by supervising a full step with the
result of two of its own half-steps (see `shortcut_forcing_step` and its
bootstrap rows in open-dreamer's `dreamer/training.py`). That is how you
push toward K = 1 without a separate distillation phase. It is the natural
follow-up project once this stage's model works.

## Design choices and their whys

Latent space, not pixels. The dynamics model runs on an 8x8x8 latent grid
(64 tokens of dim 8) from a small deterministic conv AE (`s4_latent_ae.py`,
mirroring stage 0's ConvVAE minus the variational part; determinism is fine
because flow matching supplies its own noise). Diffusion in pixel space
works but wastes model capacity on texture. Failure mode to watch: if the
AE is undertrained, the dynamics model faithfully predicts mush, and no
amount of dynamics training fixes it. Check reconstructions first.

Latent normalization. Latents are normalized per channel to zero mean and
unit variance before dynamics training, because the flow interpolates
against unit Gaussian noise; if data variance is much smaller than 1, tau
stops meaning "fraction of signal". open-dreamer does the same with dataset
stats.

Token layout. Per frame: one action token, one tau token, 64 latent tokens.
The tau token, built from a sinusoidal embedding of tau, is how the model
knows how much to trust each frame; space attention lets every latent token
read it. The action token rides along the same way. Failure mode: blind the
tau token and the denoising objective gets much harder, which shows up
directly as a roughly doubled training loss (Lab C). On this easy
environment rollout quality largely hides it, because context nearly
determines the next frame; on harder data it does not.

Factorized attention. Space layers attend among the 66 tokens of one frame;
time layers attend causally across frames at the same token position; the
two alternate through the depth. Full attention over T * 66 tokens would be
quadratically expensive and is not needed. The causal time mask is also what
makes one training pass supervise every frame at once. Failure mode: a leak
in the time mask lets frame t see frame t+1 and training loss looks
fantastic while rollouts are garbage; the perturbation test in the test file
exists for exactly this.

Uniform continuous tau. open-dreamer samples tau on a discrete grid because
shortcut distillation needs aligned half-steps. We do not implement the
bootstrap loss, so plain uniform sampling is simpler and works.

Euler mixing with fresh context noising each frame. The context is re-noised
once per generated frame (not once per ladder step), matching open-dreamer.
Re-noising every ladder step would make the context flicker under the model
mid-denoise for no benefit.

## Map to open-dreamer

The open-dreamer repo (github.com/next-state/open-dreamer, cloned next to
this repo as `../open-dreamer`) is the grown-up version of this stage.
Concept by concept:

- `flow.py` (per-frame tau sampling, interpolation, weighted x-space loss)
  is `dreamer/training.py`: `sample_tau_for_step` samples per-frame signal
  levels, `shortcut_forcing_step` corrupts latents with
  `z_tilde = (1 - sigma) * z0 + sigma * z1` and applies `loss_weight`
  (same none / ramp / v_space schemes). Their extra machinery (discrete tau
  grids, bootstrap rows, OT coupling) belongs to shortcut distillation,
  which we skip.
- `sampling4.py`'s tau ladder and context noising are
  `dreamer/generation.py`: `DenoiseSchedule` precomputes the tau values and
  the same beta_s = (1 - tau_{s+1}) / (1 - tau_s) mixing coefficients plus
  tau_ctx, and `next_latent` runs the ladder for one new frame with the
  decoded context noised to tau_ctx.
- `s4_model.py`'s factorized space and causal-time attention is
  `dreamer/models.py`: `BlockCausalTransformer`, whose layers switch between
  a space mask and a causal time mask by layer index (they place one time
  layer every four; we alternate one to one).

## Backward and forward

Backward: stage 0 built the pixel-to-latent compressor this stage's AE
mirrors. Stage 3 built the causal transformer over latent tokens and
exposed the drift problem with its PSNR curve; this stage keeps the
environment, the 64x64 frames, and the drift metric so the curves are
directly comparable, and changes only how noise enters training. Forward:
shortcut distillation for K = 1 sampling, and putting a policy inside the
dream, which is the open-dreamer program in full.

## You get it when

1. What does z_tau = (1 - tau) * noise + tau * x give you at tau = 0, at
   tau = 1, and why does the model need to be told tau at all?
2. Why does training with an independent tau per frame prepare the model
   for rolling out on its own generations?
3. Your context frames at inference are generated, so they contain errors.
   Why is re-noising them to tau 0.9 better than passing them as tau 1?
4. Derive the relation between x-space MSE and v-space MSE for the linear
   path. Where does 1 / (1 - tau)^2 come from?
5. Why does the tau ladder with K = 1 reduce to a single forward pass, and
   what exactly does beta_s = (1 - tau_{s+1}) / (1 - tau_s) preserve?
6. Stage 3 also conditioned on actions and history. State precisely what
   changed in the training distribution here, in one sentence.
7. Why can one training pass supervise all T frames at once, and which
   architectural property makes that valid?
8. Why does real-time use care about K, and what would shortcut
   distillation buy you?
