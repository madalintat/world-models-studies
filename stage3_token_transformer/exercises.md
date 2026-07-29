# Stage 3 exercises

Commit to an answer in writing before running anything. The point is calibrating your intuition, not getting it right.

All commands run from the repo root. The smoke run caches its data, so repeated runs are fast after the first.

## Prediction exercises

### P1: where does the drift curve bend?

The smoke run writes `data/stage3_token_transformer/runs/smoke/drift.csv` (PSNR per horizon step for a 5-frame rollout after 4 context frames). Before running, write down: at which horizon step does PSNR first drop by more than 2 dB from step 1, and what will the rough PSNR values be at steps 1 and 5?

```
uv run python -m stage3_token_transformer.train --smoke
cat data/stage3_token_transformer/runs/smoke/drift.csv
```

Expected observation: with a micro-trained model, PSNR is already mediocre at step 1 (roughly 15-25 dB, dominated by the quantization floor of a barely trained VQ-VAE), and over only 5 frames the curve is often flat or non-monotone; the 2 dB drop you predicted may not show up at all. That is itself the lesson at this scale: the quantization floor swamps the drift signal. The drift mechanism you are measuring is exposure bias: step 1 conditions on real context only, step 2 conditions on one sampled frame, step 5 conditions on four of them, and errors in the input distribution compound. At the full config, with a well-trained tokenizer and a 16-frame horizon, the curve bends down clearly and does not recover.

### P2: does more context delay the bend?

Same setup, but 6 context frames instead of 4. Commit first: will the drift curve at each horizon step be better, worse, or the same, and why would extra context matter at all when the model already saw 4 frames?

```
uv run python -m stage3_token_transformer.train --smoke --phase dyn --context 6
cat data/stage3_token_transformer/runs/smoke/drift.csv
```

Expected observation: usually a modest lift at early horizons, but at smoke scale a difference of a dB or two between the two settings is not meaningful (runs are seeded, so rerunning the same command reproduces the same numbers; the noise here is seed-level, not run-level). The mechanism to internalize: more real context means the KV cache holds more in-distribution evidence about track curvature and speed, so the first sampled frame is more accurate, so the second frame's input is less off-distribution, and so on. Context delays the bend; it cannot remove it, because after the context runs out the model is on its own outputs either way.

### P3: how many codes does a barely trained codebook use?

Before running the VQ phase, commit to a number: out of 256 codes, how many will have nonzero usage after 150 tiny training steps? Fewer than 20, 20-100, or more than 100?

```
uv run python -m stage3_token_transformer.train --smoke --phase vq
sort -t, -g data/stage3_token_transformer/runs/smoke/codebook_usage.csv | tail -5
```

Watch the `active N/256` counter in the training log too. Expected observation: with EMA plus dead-code reinit the active count is high (typically well over 200, since reinit keeps teleporting unused codes onto real data). The usage histogram is still far from uniform: CarRacing frames are mostly grass and road, and a handful of codes soak up most of the mass. Perplexity around 20-80 at this scale is normal and much lower than the active-code count; know the difference between "was picked at least once" and "carries real probability mass".

### P4: what does the token loss converge toward?

Cross entropy over 256 classes starts near ln(256) = 5.55 for a random model. Before running dynamics training, commit: where will the loss be after 120 smoke steps: above 5, 3-5, 1-3, or below 1?

```
uv run python -m stage3_token_transformer.train --smoke
```

Expected observation: the `[dyn]` lines drop from about 5.5 to roughly 2-4 with accuracy in the 0.2-0.5 range. Two frames of CarRacing are nearly identical, so a large fraction of tokens are predictable by copying, and even a micro-model learns that fast. The remaining loss lives in the cells that actually change: the car, the track edge ahead. This is the general shape of video-token losses: a large cheap "copy the background" component and a small expensive "predict the change" component.

## Break-it labs

### B1: kill the codebook stabilizers

Sabotage: train the VQ phase with EMA updates and dead-code reinit disabled. The codebook stays at its random initialization while the encoder trains against it.

```
uv run python -m stage3_token_transformer.train --smoke --phase vq --no-ema --no-dead-reinit
sort -t, -g data/stage3_token_transformer/runs/smoke/codebook_usage.csv | tail -5
```

What to observe: the `active N/256` count and perplexity in the log, and the usage csv afterward. Compare against the healthy run from P3. Expected: far fewer active codes (often a few dozen), perplexity in the single digits or low tens, and the top few histogram entries holding most of the mass; recon loss also stalls higher. What it teaches: nearest-neighbor assignment is a rich-get-richer process. Codes near the encoder's output distribution get picked; without EMA the picked codes do not even improve, and without reinit the unpicked ones never get a second chance. The quantizer only works because two explicit mechanisms fight the collapse the objective does nothing to prevent.

Important: this run overwrites `vqvae.pt` with the collapsed codebook, and every later `--phase dyn` command loads that file. Re-run the clean VQ phase (the P3 command, without the sabotage flags) before moving on to B2.

### B2: temperature 0 rollouts

Sabotage: sample the rollout greedily instead of from the distribution.

```
uv run python -m stage3_token_transformer.train --smoke --phase dyn --temperature 0
uv run python -m stage3_token_transformer.train --smoke --phase dyn --temperature 1
```

What to observe: open `data/stage3_token_transformer/runs/smoke/rollout.gif` after each run (generated frames on the left, ground truth on the right). Run temperature 0 twice and confirm the GIF is byte-identical, then compare its motion against temperature 1. (Training is seeded, so the temperature 1 run also reproduces on rerun; the difference is that the argmax rollout would be deterministic even with the seed removed, because it never consults the RNG.) Expected: the greedy rollout is frozen or near-frozen (each frame the argmax continuation of a mostly static scene is the same scene), while temperature 1 shows changes, at smoke scale often noisy or garbled ones. What it teaches: the model outputs a distribution over futures, and argmax collapses it to the single most probable future, which for video is overwhelmingly "nothing moves". Motion lives in the tails. This is the video version of greedy decoding making a language model repeat itself in loops.

### B3: unmask the action positions

Sabotage: in `s3_transformer.py`, edit `make_targets` so action slots are trained on too. Replace the targets line so the position before each action predicts token 0 instead of being ignored, for example by removing the IGNORE_INDEX fill and setting `seq[:, :, 0] = 0` before flattening.

```
uv run python -m stage3_token_transformer.train --smoke --phase dyn
```

What to observe: the `[dyn]` loss and accuracy compared to a clean run. Expected: loss sits visibly higher and accuracy lower, because one position in every 65 now demands predicting a fake constant that the frame content does not determine, and the gradient from those positions is pure noise for the world model. What it teaches: in interleaved sequences, what you mask is part of the model definition. The world model's job is p(next frame tokens | past, actions), not p(actions | past); leaking the controller's job into the loss dilutes the objective. Revert the edit afterward and confirm the clean loss returns.
