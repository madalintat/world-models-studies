# Stage 4 exercises

All commands run from the repo root. Every smoke run prints the first and last
PSNR of the drift curve and writes the full curve to `drift.csv` plus a
side-by-side gif (`rollout.gif`, ground truth left, rollout right) in its
output directory, `data/stage4_diffusion_forcing/out_smoke/` by default. Reruns
overwrite that directory, so whenever an exercise compares two runs the
commands pass `--out` to keep both results. Smoke numbers are noisy: a few
hundred training steps on 260 frames. Judge slopes and relative differences,
not absolute quality. If two variants land within about 1 dB of each other,
rerun before concluding anything.

## Prediction exercises

Commit to an answer in writing before running the command.

### 1. Context noise level at inference: tau_ctx 0.9 vs 0.5

Predict: the model was trained with context at every tau, so both work, but
which gives the better drift curve? At tau_ctx 0.5 the context is half noise:
does the extra noise help (washes out generation errors harder) or hurt
(destroys real signal the model needs)?

    uv run python -m stage4_diffusion_forcing.train --smoke --tau-ctx 0.9 \
        --out data/stage4_diffusion_forcing/out_tau09
    uv run python -m stage4_diffusion_forcing.train --smoke --tau-ctx 0.5 \
        --out data/stage4_diffusion_forcing/out_tau05

Expected: tau_ctx 0.5 is worse, typically 2 to 3 dB lower by the last generated
frame (a reference smoke run gave 23.0 vs 20.0 dB) and visibly less tied to the
prefill in the gif. There is a tradeoff, and 0.5 throws away too much genuine
context signal; the rollout starts ignoring its own history and regresses
toward a generic plausible road. 0.9 removes fine errors while keeping the
geometry. This is why open-dreamer defaults tau_ctx to 0.9, not to something
symmetric like 0.5.

### 2. Ladder steps: K = 4 vs K = 1

Predict: how much does dropping to a single step cost in PSNR, given the model
was trained as an x-predictor at every tau, including tau = 0?

    uv run python -m stage4_diffusion_forcing.train --smoke --k-steps 4 \
        --out data/stage4_diffusion_forcing/out_k4
    uv run python -m stage4_diffusion_forcing.train --smoke --k-steps 1 \
        --out data/stage4_diffusion_forcing/out_k1

Expected: K = 1 works (that surprises most people: at tau = 0 the model is
literally trained to map pure noise plus context to a clean frame). On this
environment at smoke scale the two curves land within run-to-run noise of each
other, and K = 1 can even come out ahead: a one-shot x-prediction is the
conditional mean, and PSNR rewards conservative, averaged predictions. The cost
of K = 1 is variance and detail you can barely see here because consecutive
CarRacing frames are so predictable; on harder data the gap grows, which is
what motivates shortcut distillation instead of just always using K = 1.

### 3. Loss weighting: where does the training signal go

Predict: with `--weighting none`, which taus dominate the loss value, high or
low? Write down the expected loss magnitude at tau near 0 versus tau near 1
before running (hint: at tau near 0 the model has almost no signal, so its best
x-prediction error approaches the data variance, which is 1 after
normalization).

    uv run python -m stage4_diffusion_forcing.train --smoke --weighting none

Watch the printed loss values against a `--weighting ramp` run. Expected: with
none, near-noise frames, where the error is inherently around 1 and mostly
irreducible, get equal say in the average, so the printed loss sits somewhat
higher than ramp's (the loss normalizes by the weight sum, which keeps the
scales comparable and the shift modest at smoke scale). The point survives the
small numbers: the gradient budget under none is spent where it matters least,
and ramp and v_space shift it toward high tau, where the last ladder steps
live.

### 4. The drift curve shape

Before your first smoke run, sketch the PSNR-vs-t curve you expect over the 12
generated frames. Stage 3's curve fell steadily. Predict: falling, flat, or
falling then flat?

    uv run python -m stage4_diffusion_forcing.train --smoke
    cat data/stage4_diffusion_forcing/out_smoke/drift.csv

Expected: it starts near the AE's own reconstruction ceiling (high 20s dB at
smoke scale, since the first generated frame leans on a clean ground-truth
prefill) and gives up a few dB over the 12 frames; a reference run went from
28.8 to 23.0 dB. Part of that slope is not drift at all: the AE reconstructs
later frames of this episode a couple of dB worse even from ground-truth
latents. The instructive comparison is Lab A: the sabotaged clean-context model
sits 7+ dB lower from the very first generated frame, because errors it was
never trained to see compound immediately, while the diffusion-forced model
degrades gently.

### 5. Action injection: does the mechanism matter here?

The action reaches the dynamics model as an extra token in each frame's space
sequence. That is a choice, not a law. `--injection additive` adds the action
embedding to every latent token instead, costing no sequence slot and no
parameters; `--injection film` gives every block a per-frame scale and shift
predicted from the action.

nano-world-model ran this ablation properly, at scale, with five mechanisms.
Before looking anything up, commit to two answers. First: for CarRacing's
3-number action, will the three mechanisms land within about 1 dB of each
other, or will one clearly win? Second: which one would you bet on, and is
your reason about capacity or about optimization?

    uv run python -m stage4_diffusion_forcing.train --smoke --injection token \
        --out data/stage4_diffusion_forcing/out_inj_token
    uv run python -m stage4_diffusion_forcing.train --smoke --injection additive \
        --out data/stage4_diffusion_forcing/out_inj_additive
    uv run python -m stage4_diffusion_forcing.train --smoke --injection film \
        --out data/stage4_diffusion_forcing/out_inj_film

Expected: they cluster. Measured reference, same seed:

| Injection | Params | Last-frame PSNR |
|:--|--:|--:|
| `token` | 214472 | 23.18 dB |
| `additive` | 214408 | 22.80 dB |
| `film` | 247688 | 22.09 dB |

A 1.1 dB spread across the three, on a single seed, at smoke scale, is not a
ranking. Resist reading one. The parameter column is the part that is real:
additive is 64 parameters cheaper than token and costs no sequence slot,
while film pays 33k parameters for a modulation head in every block.

This matches the published result rather than contradicting it, which is the
point of the exercise. nano-world-model's PushT sweep (2D actions) put all
five mechanisms within 0.32 PSNR, with plain additive winning at zero extra
cost; only on 7D robot actions did FiLM pull ahead, and cross-attention was
consistently worst everywhere. The lesson to take is about when an
architectural knob is worth spending on: a 3-number action carries so little
information that any path into the network suffices, and the interesting
question was never "which is best" but "how much does this axis matter at
all". Be suspicious of papers that ablate a knob without telling you the
regime in which the knob is dead.

## Break-it labs

### Lab A: clean context (the money experiment)

Sabotage: train with all context frames clean at tau = 1 and only the last
frame noised and supervised, exactly stage 3's teacher forcing recipe, then
roll out.

    uv run python -m stage4_diffusion_forcing.train --smoke --clean-context \
        --out data/stage4_diffusion_forcing/out_clean_ctx
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --out data/stage4_diffusion_forcing/out_forcing

(The first command automatically rolls out with tau_ctx = 1.0 to match its own
training distribution.) Compare the two drift.csv files and gifs.

Observe: at smoke scale the clean-context run does not degrade gracefully, it
collapses outright. A reference run put it around 15 dB, roughly flat, from the
very first generated frame, against 28.8 falling to 23.0 for diffusion forcing,
and its gif smears immediately. Two confounds, disclosed so you self-check
honestly: the recipe supervises one frame per sequence instead of eight (8x
less training signal), and that frame always sits at the last position of a
full window, so the shorter contexts met at rollout were never trained either.
Both are what teacher forcing costs you in this codebase, but they mean the
smoke gap overstates the pure train-test-mismatch effect; at full scale the
mismatch effect survives equal-compute corrections.

Teaches: train-test mismatch in the context, not model capacity, is what caused
stage 3's drift. One line of tau sampling is the fix.

### Lab B: weighting none vs v_space

Sabotage: remove all loss weighting, then overcorrect to v_space.

    uv run python -m stage4_diffusion_forcing.train --smoke --weighting none \
        --out data/stage4_diffusion_forcing/out_wnone
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --weighting v_space --out data/stage4_diffusion_forcing/out_wvspace

Observe: none lands close to ramp on the drift curve (within about 1 dB in
reference runs, judge across reruns). v_space is the interesting one: at smoke
scale it visibly destabilizes the short run. Its printed loss swings by an
order of magnitude between prints (the weights near tau = 1 are huge, tamed
only by the eps clamp) and its rollout lands several dB below ramp's; a
reference run started at 17.8 dB instead of 28.8. The reliable smoke observable
is the loss trace: spiky for v_space, smooth for none and ramp. v_space's
payoff, sharper high-tau detail, needs the long full-scale run to show up,
which is why the full config recommends it and the smoke default is ramp.

Teaches: the weighting does not change what the model can express, only where
its training signal concentrates on the tau axis, and that is exactly the
x-pred vs v-pred difference.

### Lab C: blind the tau token

Sabotage: in `s4_model.py`, at the top of `DynamicsTransformer.forward`, add
`tau = torch.full_like(tau, 0.5)`, so the model is never told each frame's true
noise level, in training or at rollout (the loss still uses the real tau).
Retrain:

    uv run python -m stage4_diffusion_forcing.train --smoke \
        --out data/stage4_diffusion_forcing/out_blind_tau

Observe: the training loss, not the drift curve, is where the damage shows. At
equal steps the printed loss runs about twice as high as the honest run's
(reference: 0.59 vs 0.23 at step 120), because guessing how much to trust each
frame is strictly harder than being told. The drift curve barely moves, under 1
dB in reference runs, and that is the second lesson: consecutive CarRacing
frames are so predictable that context alone nearly determines the next frame,
so the end metric hides a real conditioning defect that the loss exposes.
Revert the change afterward.

A softer variant, no retraining needed: in `sampling4.py`, in `rollout`, make
`ctx_tau` label generated frames 1.0 while still actually noising them, and
rerun the trainer. Commit to a prediction first. Measured honestly: at smoke
scale this also moves PSNR by well under 1 dB. If you predicted a collapse,
notice what that teaches you about trusting a mechanism story without measuring
it.

### Lab D: build the pyramid schedule, then price the corners

This one is a build, not a sabotage, and it is the piece of this stage worth
writing yourself. `flow.py` ships `_full_sequence_schedule` and
`_sequential_schedule` but raises `NotImplementedError` for
`_pyramid_schedule`. The docstring there gives you the exact contract, a
worked example for K = 2 with three frames, and the reason `stagger` is a
knob rather than a constant. It is about six lines. Three tests in
`tests/test_stage4_diffusion_forcing.py` currently skip and will start
running the moment you write it, including one that checks stagger 0 recovers
`full_sequence` and stagger K recovers `sequential`.

Before writing it, commit to an answer: over a four-frame block, how many
model calls does each mode need at K = 4? Then check yourself:

    uv run python -m stage4_diffusion_forcing.train --smoke \
        --schedule full_sequence \
        --out data/stage4_diffusion_forcing/out_sched_full
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --schedule pyramid \
        --out data/stage4_diffusion_forcing/out_sched_pyramid
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --out data/stage4_diffusion_forcing/out_sched_seq

Each run prints its model-call count. Measured reference, same four-frame
block, same seed:

| Mode | Model calls | Last-frame PSNR |
|:--|--:|--:|
| `full_sequence` | 4 | 27.27 dB |
| `pyramid` (stagger 1) | 7 | 28.15 dB |
| `sequential` | 16 | not run here |

The `sequential` row is arithmetic, not a measurement: K * n_frames = 16, and
the default 12-frame autoregressive rollout costs 48, which is why the block
modes print a note about capping the horizon to the attention window.

Observe the shape of the trade rather than the winner. Pyramid buys about 0.9
dB on the last frame for 3 extra model calls, so the dial is real and it
points the way you would expect: more calls, better conditioning, better
frames. But notice how flat it is. Quadrupling the calls from 4 to 16 cannot
plausibly buy four times the 0.9 dB, because on four frames from a clean
ground-truth prefill there is very little for the conditioning to disagree
about in the first place. The regime where the expensive corner earns its
money is long blocks where a frame genuinely needs its predecessor resolved
before it can commit, which is exactly what the sliding-window rollout and its
re-noised context exist to handle.

Teaches: diffusion forcing's training-time trick (independent per-frame tau)
buys an inference-time freedom, and that freedom is a cost/coherence dial you
choose per deployment rather than a property of the trained model.

### Lab E: logit-normal tau sampling

Sabotage is the wrong word again: this is the SD3 default that
nano-world-model uses, and the question is whether it earns its keep here.
Uniform tau spends equal training budget on every noise level. Logit-normal
draws u ~ N(0, 1) and uses sigmoid(u), concentrating on tau near 0.5.

Commit first: does concentrating the budget on mid-noise levels help the drift
curve, hurt it, or do nothing measurable on CarRacing?

    uv run python -m stage4_diffusion_forcing.train --smoke \
        --tau-sampling logit_normal \
        --out data/stage4_diffusion_forcing/out_tau_logitnormal
    uv run python -m stage4_diffusion_forcing.train --smoke \
        --out data/stage4_diffusion_forcing/out_tau_uniform

Expected at smoke scale: nothing you can distinguish from run-to-run noise (a
reference pair gave 22.45 dB for logit-normal against 23.18 dB for uniform on
the last frame, so logit-normal is slightly behind on one seed, which is to
say nothing at all). Do not read that as "SD3 is wrong". Read the shape of the
argument: the recipe exists because at scale, on hard data, the ends of the
tau range are wasted budget. At K = 4 on a 260-frame CarRacing smoke set,
there is no budget pressure to relieve. A technique can be correct and still
be unmeasurable in your regime, and knowing which regime you are in is worth
more than knowing the technique.

Teaches: how to read a defaults table from someone else's repo. Their default
is evidence about their regime, not a universal ranking.

Teaches: the tau token is what makes the denoising objective well posed, and
the loss curve proves the model uses it. Whether a conditioning defect shows up
in your end metric depends on how much the data lets the model lean on other
signals; on harder, faster-changing data it would.
