# Stage 4 exercises

All commands run from the repo root. Every smoke run prints the first and
last PSNR of the drift curve and writes the full curve to `drift.csv` plus a
side-by-side gif (`rollout.gif`, ground truth left, rollout right) in its
output directory, `data/stage4_diffusion_forcing/out_smoke/` by default.
Reruns overwrite that directory, so whenever an exercise compares two runs
the commands pass `--out` to keep both results. Smoke numbers are noisy:
a few hundred training steps on 260 frames. Judge slopes and relative
differences, not absolute quality. If two variants land within about 1 dB
of each other, rerun before concluding anything.

## Prediction exercises

Commit to an answer in writing before running the command.

### 1. Context noise level at inference: tau_ctx 0.9 vs 0.5

Predict: the model was trained with context at every tau, so both work,
but which gives the better drift curve? At tau_ctx 0.5 the context is half
noise: does the extra noise help (washes out generation errors harder) or
hurt (destroys real signal the model needs)?

    uv run python -m stage4_diffusion_forcing.train --smoke --tau-ctx 0.9 --out data/stage4_diffusion_forcing/out_tau09
    uv run python -m stage4_diffusion_forcing.train --smoke --tau-ctx 0.5 --out data/stage4_diffusion_forcing/out_tau05

Expected: tau_ctx 0.5 is worse, typically 2 to 3 dB lower by the last
generated frame (a reference smoke run gave 23.0 vs 20.0 dB) and visibly
less tied to the prefill in the gif. There is a tradeoff, and 0.5 throws
away too much genuine context signal; the rollout starts ignoring its own
history and regresses toward a generic plausible road. 0.9 removes fine
errors while keeping the geometry. This is why open-dreamer defaults
tau_ctx to 0.9, not to something symmetric like 0.5.

### 2. Ladder steps: K = 4 vs K = 1

Predict: how much does dropping to a single step cost in PSNR, given the
model was trained as an x-predictor at every tau, including tau = 0?

    uv run python -m stage4_diffusion_forcing.train --smoke --k-steps 4 --out data/stage4_diffusion_forcing/out_k4
    uv run python -m stage4_diffusion_forcing.train --smoke --k-steps 1 --out data/stage4_diffusion_forcing/out_k1

Expected: K = 1 works (that surprises most people: at tau = 0 the model is
literally trained to map pure noise plus context to a clean frame). On this
environment at smoke scale the two curves land within run-to-run noise of
each other, and K = 1 can even come out ahead: a one-shot x-prediction is
the conditional mean, and PSNR rewards conservative, averaged predictions.
The cost of K = 1 is variance and detail you can barely see here because
consecutive CarRacing frames are so predictable; on harder data the gap
grows, which is what motivates shortcut distillation instead of just always
using K = 1.

### 3. Loss weighting: where does the training signal go

Predict: with `--weighting none`, which taus dominate the loss value, high
or low? Write down the expected loss magnitude at tau near 0 versus tau
near 1 before running (hint: at tau near 0 the model has almost no signal,
so its best x-prediction error approaches the data variance, which is 1
after normalization).

    uv run python -m stage4_diffusion_forcing.train --smoke --weighting none

Watch the printed loss values against a `--weighting ramp` run. Expected:
with none, near-noise frames, where the error is inherently around 1 and
mostly irreducible, get equal say in the average, so the printed loss sits
somewhat higher than ramp's (the loss normalizes by the weight sum, which
keeps the scales comparable and the shift modest at smoke scale). The point
survives the small numbers: the gradient budget under none is spent where
it matters least, and ramp and v_space shift it toward high tau, where the
last ladder steps live.

### 4. The drift curve shape

Before your first smoke run, sketch the PSNR-vs-t curve you expect over the
12 generated frames. Stage 3's curve fell steadily. Predict: falling,
flat, or falling then flat?

    uv run python -m stage4_diffusion_forcing.train --smoke
    cat data/stage4_diffusion_forcing/out_smoke/drift.csv

Expected: it starts near the AE's own reconstruction ceiling (high 20s dB
at smoke scale, since the first generated frame leans on a clean
ground-truth prefill) and gives up a few dB over the 12 frames; a reference
run went from 28.8 to 23.0 dB. Part of that slope is not drift at all: the
AE reconstructs later frames of this episode a couple of dB worse even from
ground-truth latents. The instructive comparison is Lab A: the sabotaged
clean-context model sits 7+ dB lower from the very first generated frame,
because errors it was never trained to see compound immediately, while the
diffusion-forced model degrades gently.

## Break-it labs

### Lab A: clean context (the money experiment)

Sabotage: train with all context frames clean at tau = 1 and only the last
frame noised and supervised, exactly stage 3's teacher forcing recipe, then
roll out.

    uv run python -m stage4_diffusion_forcing.train --smoke --clean-context --out data/stage4_diffusion_forcing/out_clean_ctx
    uv run python -m stage4_diffusion_forcing.train --smoke --out data/stage4_diffusion_forcing/out_forcing

(The first command automatically rolls out with tau_ctx = 1.0 to match its
own training distribution.) Compare the two drift.csv files and gifs.

Observe: at smoke scale the clean-context run does not degrade gracefully,
it collapses outright. A reference run put it around 15 dB, roughly flat,
from the very first generated frame, against 28.8 falling to 23.0 for
diffusion forcing, and its gif smears immediately. Two confounds, disclosed
so you self-check honestly: the recipe supervises one frame per sequence
instead of eight (8x less training signal), and that frame always sits at
the last position of a full window, so the shorter contexts met at rollout
were never trained either. Both are what teacher forcing costs you in this
codebase, but they mean the smoke gap overstates the pure
train-test-mismatch effect; at full scale the mismatch effect survives
equal-compute corrections.

Teaches: train-test mismatch in the context, not model capacity, is what
caused stage 3's drift. One line of tau sampling is the fix.

### Lab B: weighting none vs v_space

Sabotage: remove all loss weighting, then overcorrect to v_space.

    uv run python -m stage4_diffusion_forcing.train --smoke --weighting none --out data/stage4_diffusion_forcing/out_wnone
    uv run python -m stage4_diffusion_forcing.train --smoke --weighting v_space --out data/stage4_diffusion_forcing/out_wvspace

Observe: none lands close to ramp on the drift curve (within about 1 dB in
reference runs, judge across reruns). v_space is the interesting one: at
smoke scale it visibly destabilizes the short run. Its printed loss swings
by an order of magnitude between prints (the weights near tau = 1 are huge,
tamed only by the eps clamp) and its rollout lands several dB below ramp's;
a reference run started at 17.8 dB instead of 28.8. The reliable smoke
observable is the loss trace: spiky for v_space, smooth for none and ramp.
v_space's payoff, sharper high-tau detail, needs the long full-scale run to
show up, which is why the full config recommends it and the smoke default
is ramp.

Teaches: the weighting does not change what the model can express, only
where its training signal concentrates on the tau axis, and that is exactly
the x-pred vs v-pred difference.

### Lab C: blind the tau token

Sabotage: in `s4_model.py`, at the top of `DynamicsTransformer.forward`,
add `tau = torch.full_like(tau, 0.5)`, so the model is never told each
frame's true noise level, in training or at rollout (the loss still uses
the real tau). Retrain:

    uv run python -m stage4_diffusion_forcing.train --smoke --out data/stage4_diffusion_forcing/out_blind_tau

Observe: the training loss, not the drift curve, is where the damage shows.
At equal steps the printed loss runs about twice as high as the honest
run's (reference: 0.59 vs 0.23 at step 120), because guessing how much to
trust each frame is strictly harder than being told. The drift curve barely
moves, under 1 dB in reference runs, and that is the second lesson:
consecutive CarRacing frames are so predictable that context alone nearly
determines the next frame, so the end metric hides a real conditioning
defect that the loss exposes. Revert the change afterward.

A softer variant, no retraining needed: in `sampling4.py`, in `rollout`,
make `ctx_tau` label generated frames 1.0 while still actually noising
them, and rerun the trainer. Commit to a prediction first. Measured
honestly: at smoke scale this also moves PSNR by well under 1 dB. If you
predicted a collapse, notice what that teaches you about trusting a
mechanism story without measuring it.

Teaches: the tau token is what makes the denoising objective well posed,
and the loss curve proves the model uses it. Whether a conditioning defect
shows up in your end metric depends on how much the data lets the model
lean on other signals; on harder, faster-changing data it would.
