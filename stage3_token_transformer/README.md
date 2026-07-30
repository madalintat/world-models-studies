# Stage 3: VQ-VAE tokens + autoregressive transformer (IRIS-style)

CarRacing frames become 8x8 grids of discrete codes from a 256-entry codebook; a small GPT over interleaved `[action, 64 frame tokens]` sequences predicts the next frame token by token. Rollouts sample the future autoregressively with a KV cache and measure drift (PSNR vs the ground-truth continuation).

## What's here

- `vqvae.py`: conv encoder/decoder plus `VectorQuantizerEMA` (straight-through estimator, commitment beta 0.25, EMA codebook, dead-code reinit, usage histogram). This is the file to hand-write first, the quantizer especially.
- `s3_transformer.py`: `TokenGPT`, a small causal transformer with learned positions, one linear-embedded action token per frame, loss masking on action slots, KV cache via `step()`.
- `rollout3.py`: KV-cached autoregressive rollout, PSNR drift curve, side-by-side rollout GIF.
- `s3_data.py`: CarRacing-v3 collection with a scripted driver, fixed seeds, cached under `data/stage3_token_transformer/`.
- `train.py`: `--phase vq|dyn|all`, `--smoke`, break-it flags `--no-ema`, `--no-dead-reinit`, rollout knobs `--context`, `--future`, `--temperature`.
- `WHY.md`, `exercises.md`, `tests/test_stage3_token_transformer.py`.

## Commands

```
# tests (about half a minute, self-contained, no cached data needed)
uv run pytest stage3_token_transformer/tests -q

# smoke: collects 2 short episodes, micro-trains both phases on CPU,
# writes drift.csv and rollout.gif (about 2 minutes)
uv run python -m stage3_token_transformer.train --smoke

# individual phases
uv run python -m stage3_token_transformer.train --smoke --phase vq
uv run python -m stage3_token_transformer.train --smoke --phase dyn
```

Outputs land in `data/stage3_token_transformer/runs/smoke/`: `vqvae.pt`, `gpt.pt`, `codebook_usage.csv`, `drift.csv` (PSNR per horizon step), `rollout.gif` (generated | ground truth).

## Full run on one RTX 5090

Defaults without `--smoke` are the full config; pass `--device cuda` for a real run (the rollout follows the model device automatically).

- Data: 250 episodes x 200 steps = 50k frames, collected once (~40 min, CPU-bound in Box2D).
- `--phase vq`: base 128, batch 128, 60k steps, lr 3e-4. About 1h. Expect recon MSE below 1e-3 and codebook perplexity above 100.
- `--phase dyn`: d_model 512, 8 layers, 20-frame windows (1300 tokens), batch 12, 100k steps. About 3-6h. Expect next-token accuracy of 60-80% and rollouts that hold road geometry for 20+ frames.

Modal cost: roughly $12-25 total for the 5-7 GPU hours on an A100/H100-class instance.
