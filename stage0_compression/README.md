# Stage 0: Compression (AE and VAE on CarRacing frames)

Learn why a 32-number bottleneck forces a network to understand a scene, and why the VAE's sampled, KL-regularized latent space (not the plain AE's) is the substrate the rest of the course builds on. Read `WHY.md` first, then hand-write `models.py` yourself, then do `exercises.md`.

## What's here

- `collect.py`: random-policy frame collection from CarRacing-v3, frame skip 4, resized to 64x64 uint8, cached as npz under `data/stage0_compression/`, seeded.
- `models.py`: `ConvAE` and `ConvVAE` (latent dim 32, four convs each way, World Models architecture) plus the loss functions. This is the file to write yourself first.
- `train.py`: trains either model, logs recon and KL, saves a checkpoint and a reconstruction grid png.
- `viz.py`: reconstruction grid, latent traversal, interpolation strip, each also available from the CLI against a saved checkpoint.
- `tests/test_stage0_compression.py`: shapes, loss decrease, reparam gradient flow, viz output shapes.

## Commands

Everything runs from the repo root.

```
# fast CPU smoke run (collects 120 frames on first use, ~30 s total)
uv run python -m stage0_compression.train --smoke
uv run python -m stage0_compression.train --smoke --model ae
uv run python -m stage0_compression.train --smoke --beta 0.0

# visualize a checkpoint
uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_smoke.pt --mode grid
uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_smoke.pt --mode traversal --dims 0 1 2 3
uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_smoke.pt --mode interp

# tests
uv run python -m pytest stage0_compression/tests -q
```

Outputs land in `runs/stage0_compression/` (gitignored, like `data/`).

## Full run on one RTX 5090

```
uv run python -m stage0_compression.collect --episodes 40 --frames-per-episode 500 --seed 0
uv run python -m stage0_compression.train --model vae --device cuda
```

20k frames, batch 128, 30k steps, Adam 1e-4 (these are the defaults). Expect 30 to 45 minutes wall clock; collection itself is CPU-bound and takes a few minutes. Outcome: recon loss around 20 to 40 (summed MSE per frame), KL around 30 to 60 nats, reconstructions with clear road geometry and car position but smoothed textures. Train the AE with `--model ae` for the comparison exercises.

Modal cost: one 5090/A100-class GPU for under an hour, roughly 2 to 4 USD per full training run.
