# Stage 0 exercises

Rules: for prediction exercises, write your answer down before running the command. The point is to catch your own model of the model being wrong. All commands run from the repo root. Smoke runs take under a minute each on CPU.

## Prediction exercises

### P1. Reconstruction quality at z=2 vs z=32

Predict first: with a 2-dim latent, what survives in the reconstructions and what disappears? Rank these: road curvature, car position, grass/road boundary, curb stripes. Then run both and compare the recon grids side by side.

```
uv run python -m stage0_compression.train --smoke --latent-dim 2 --tag vae_z2
uv run python -m stage0_compression.train --smoke --latent-dim 32 --tag vae_z32
```

Open `runs/stage0_compression/vae_z2_recon.png` and `vae_z32_recon.png`.

Expected: z=2 reconstructions converge toward one or two generic "average track" images; the grass/road split survives, but car position and specific curvature are mostly gone. z=32 keeps road shape and car position per frame. Curb stripes are blurry in both. If the difference looks small at 150 smoke steps, that is itself informative: early in training even z=32 has not used its capacity yet, and the gap widens with more steps.

### P2. AE vs VAE reconstruction loss

Predict first: after the same number of steps, which model has lower reconstruction error, the AE or the VAE at beta=1.0? By roughly what factor?

```
uv run python -m stage0_compression.train --smoke --model ae
uv run python -m stage0_compression.train --smoke --model vae
```

Compare the final `recon` numbers in the logs.

Expected: the AE wins on reconstruction, usually by a modest margin (tens of percent, not orders of magnitude). It should win: it pays no KL tax and decodes exact points instead of noisy samples. The VAE is deliberately trading reconstruction for a usable latent geometry. If your AE is not beating your VAE on recon, something is off in the VAE-favoring direction (check beta).

### P3. Where does the KL curve go

Predict first: sketch the KL value over the 150 smoke steps. Does it start high and fall, start near zero and rise, or something else?

```
uv run python -m stage0_compression.train --smoke --model vae
```

Watch the `kl` column.

Expected: it starts near zero (a freshly initialized encoder outputs mu near 0, logvar near 0, which is exactly the prior), rises as the encoder starts actually using the latent to help reconstruction, and would settle into the tens of nats in a long run. KL rising early is the encoder buying information capacity and paying for it. In a short smoke run you may see it rise then dip as the KL penalty pushes back; the point is that nonzero KL is the sign of a working encoder, not a cost to be minimized to zero.

### P4. Interpolation smoothness, AE vs VAE

Predict first: decoding 8 evenly spaced points on the line between two frame codes, which model gives frames that all look like plausible track scenes, and which gives murky in-between images?

```
uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/ae_smoke.pt --mode interp
uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_smoke.pt --mode interp
```

Open the two `*_interp.png` strips.

Expected: the VAE strip morphs through images that each individually look like a (blurry) track. The AE strip tends to cross-fade: ghostly double roads, midpoints that look like two frames overlaid rather than a scene. At smoke scale the gap is visible but not dramatic; it becomes stark with full training. This is the "holes in the latent space" argument made visible, and it is the single most important picture in this stage.

## Break-it labs

### B1. beta = 0: the VAE degenerates into an AE

Sabotage:

```
uv run python -m stage0_compression.train --smoke --beta 0.0 --tag vae_beta0
uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_beta0.pt --mode interp
uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_beta0.pt --mode traversal --dims 0 1 2 3 --span 3.0
```

What to observe: recon loss drops a bit faster than at beta=1.0 and KL climbs without anything restraining it (watch it keep growing in the log). The interpolation strip picks up AE-style artifacts. The traversal is the sharper diagnostic: at beta=0 the codes are not calibrated to any shared scale, so sweeping a dim over [-3, 3] either does nothing visible or veers into garbage, because those coordinates mean nothing away from where the data happens to sit.

What it teaches: the reparameterized sampling alone does not give you a usable space; the KL term is what pins the code distribution to a known place and scale. beta=0 with sampling is nearly an AE with noise injection.

### B2. beta = 50: posterior collapse on demand

Sabotage:

```
uv run python -m stage0_compression.train --smoke --beta 50.0 --tag vae_beta50
```

What to observe: the `kl` column gets crushed toward zero within tens of steps and stays there, while recon loss plateaus far above the beta=1.0 run. Open `runs/stage0_compression/vae_beta50_recon.png`: the bottom row is the same green-and-gray mean-image smear for every input. The top row differs, the bottom row does not. That is the signature.

What it teaches: collapse is not exotic, it is one flag away. And it is quiet: total loss still decreases, nothing crashes. You now know the tell (KL pinned at zero, identical reconstructions) and you will recognize it when a scaling bug, not a flag, causes it.

### B3. Remove the frame skip: the too-easy dataset

Sabotage: collect a dataset with frame skip 1, train on it, and compare loss against the normal one.

```
uv run python -m stage0_compression.collect --episodes 2 --frames-per-episode 60 --frame-skip 1 --seed 0 --out data/stage0_compression/noskip.npz
uv run python -m stage0_compression.train --smoke --data data/stage0_compression/noskip.npz --tag vae_noskip
```

What to observe: the noskip run reaches a lower recon loss than the normal smoke run in the same 150 steps. Look at the recon grid: the eight sampled "different" frames are nearly the same picture. The dataset spans about 60 environment steps of actual driving instead of 240, so the model is being tested on a much smaller world.

What it teaches: loss numbers are only comparable on comparable data, and near-duplicate frames inflate apparent performance. Data diversity, not model capacity, is often the binding constraint. Keep this in mind at every later stage: whenever a change touches data collection, loss comparisons across the change are suspect.
