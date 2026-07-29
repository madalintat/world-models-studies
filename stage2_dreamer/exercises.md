# Stage 2 exercises

Commit to an answer in writing before running anything. The point is to find
out where your model of the model is wrong.

All commands run from the repo root.

## Prediction exercises

### P1: The silent KL

The smoke run prints `wm/kl` around 0.09 nats. Predict: if you rerun with the
KL balance disabled entirely, `--kl-alpha 0.0`, will `wm/total` change?

    uv run python -m stage2_dreamer.train --smoke
    uv run python -m stage2_dreamer.train --smoke --kl-alpha 0.0

Expected: identical losses to three decimals. With the KL at 0.09 nats, both
sides of the balanced KL are clamped by `FREE_NATS = 1.0`, so alpha
multiplies a constant and contributes zero gradient either way. The KL term
only starts to matter once the posterior carries more than one nat. If you
predicted "alpha changes the loss", you learned what free bits actually do.

### P2: Horizon 50

Predict what happens to `ac/img_return` and to the wall clock of the
actor-critic step when the horizon goes from 15 to 50. Write down a factor
for each.

    uv run python -m stage2_dreamer.train --smoke
    uv run python -m stage2_dreamer.train --smoke --horizon 50

Expected: the smoke run takes visibly longer (the rollout is strictly
sequential, 50 GRU steps instead of 15, so the AC phase roughly triples).
`ac/img_return` barely moves, maybe 5%: the metric averages over all states
in the rollout, the untrained reward head predicts near-zero rewards
everywhere, and with lambda * gamma = 0.947 the contribution of step k
decays like 0.947^k, so most of the extra 35 steps add almost nothing. If
you predicted "3x the horizon means roughly 3x the return", the lambda
weighting is what you were missing. In a trained model the interesting
effect is different and worse: at H = 50 the rollout leaves the region the
model knows, the reward head extrapolates, and imagined reward inflates
into fiction that the actor then optimizes. That is the real reason the
horizon stays at 15.

### P3: The size of the reconstruction loss

Before running, predict the order of magnitude of `wm/recon` at the start of
training. The loss is squared error summed over all 64 * 64 * 3 = 12288
pixel dimensions of a frame scaled to [-0.5, 0.5], averaged over batch and
time.

    uv run python -m stage2_dreamer.train --smoke

Expected: around 1000. An untrained decoder outputs roughly zero everywhere,
and real frames have per-pixel deviations around 0.25 to 0.3 from mid-gray,
so 12288 * 0.08-ish lands near 10^3. Now compare with `wm/kl` near 0.1. This
ratio is deliberate: reconstruction summed over pixels must dwarf a KL of a
few nats, otherwise the cheapest way to cut the loss is to empty the latent.

### P4: Determinism of the smoke run

Predict: if you run the smoke command twice in a row, do the printed losses
match exactly? Then predict what the second run does differently at all.

    rm -rf data/stage2_dreamer
    uv run python -m stage2_dreamer.train --smoke
    uv run python -m stage2_dreamer.train --smoke

Expected: identical numbers. The prefill episode is collected with a fixed
seed and cached under `data/stage2_dreamer/`, torch is seeded, and the
second run skips the environment entirely and loads the cached episode, so
it is faster and byte-for-byte the same. If the numbers ever diverge you
introduced hidden state; find it before trusting any ablation.

## Break-it labs

### Lab 1: Kill the prior (alpha = 0)

Sabotage: with `--kl-alpha 0.0`, the prior receives zero gradient; only the
posterior is pulled toward a frozen random predictor. Free bits hide this at
smoke scale (see P1), so make the world model actually learn: in
`smoke_config()` in `train.py`, set `wm_steps_per_iter=300` and
`batch_size=8`, then run both:

    uv run python -m stage2_dreamer.train --smoke
    uv run python -m stage2_dreamer.train --smoke --kl-alpha 0.0

Before running, commit to a prediction for `wm/kl` and `wm/recon` at the end
of each run. The naive story says the KL gap should blow up when the prior
stops learning. Expected (our measurements at these settings): recon ends
around 99 with alpha 0.0 against 81 with the default, and `wm/kl` ends
LOWER with alpha 0.0, roughly 2.7 against 6.9. Both follow from one fact:
alpha decides who moves to close the gap. At 0.0 nothing else in the loss
touches the prior network, so the predictor is frozen at initialization,
and the entire KL burden lands on the posterior, which gets dragged toward
a random predictor. Perception is crippled (worse recon) and the latent
carries less information (lower kl), yet dreaming is still broken, because
every imagined step samples from a predictor that never learned anything.
What it teaches: the prior is the only thing you can dream with, and the
0.8/0.2 split exists so the predictor does the walking while perception
stays sharp. Revert the config edit afterward.

### Lab 2: Deterministic latents

Sabotage: `--det-z` replaces sampling with argmax, so z is a deterministic
function of the logits. Run the same lengthened smoke config as Lab 1:

    uv run python -m stage2_dreamer.train --smoke --det-z

Then check rollout diversity directly:

    uv run python -m stage2_dreamer.s2_dream_diversity

Expected: the helper prints that two imagined steps from the same state are
identical under det-z and different under sampling. In the training run,
det-z ends visibly worse on both metrics (in our measurements recon around
115 against 81, kl around 1.6 against 6.9). Two lessons stacked. First,
sampling is not just about modeling uncertainty; it is also exploration
over the 16 classes of each variable. Argmax always selects the current
winner, the other classes never receive a straight-through gradient, and
capacity locks in early. Second, deterministic z means every dream from a
given state is the same dream: the actor never sees the model's
uncertainty and learns an overconfident policy that the real environment
then falsifies. A model that is both weaker and more confident is the
worst combination you can hand a policy.

### Lab 3: Cut the straight-through wire

Sabotage: in `rssm.py`, in `RSSM.sample`, replace

    z = onehot + probs - probs.detach()

with

    z = onehot

Keep the lengthened smoke config from Lab 1 and run:

    uv run python -m stage2_dreamer.train --smoke

Expected: the run still executes and every loss is still finite, which is
the trap: nothing crashes when you cut a gradient path. But recon ends far
worse (in our measurements around 115 against 81) and `wm/kl` collapses to
near 0.2 instead of growing toward 7. With the straight-through term gone,
the reconstruction loss can no longer reach the posterior logits through z,
so nothing asks the latent to carry information about the frame. The only
gradient left on the posterior is the KL pull toward the prior, so the
posterior obediently empties itself and kl falls to the floor while the
decoder struggles along on h alone. What it teaches: those two odd-looking
terms `+ probs - probs.detach()` are the entire reason a discrete latent is
trainable by backprop. Also a diagnostic habit: a KL that collapses to near
zero while reconstruction stalls means information has stopped flowing into
z, and the first thing to check is the gradient path. Revert the edit.
