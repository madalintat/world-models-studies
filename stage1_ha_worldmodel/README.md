# Stage 1: Ha's world model on CarRacing (V + M + C)

The 2018 "World Models" recipe: a ConvVAE compresses frames to z (32 dims),
an MDN-RNN predicts the next z from (z, a), and an 867-parameter linear
controller is optimized with CMA-ES inside the learned model's dream.

## Files

- `WHY.md`: read first. Hand-write `mdnrnn.py` before reading the one here.
- `s1_vae.py`: ConvVAE, z = 32 (same model as stage 0).
- `mdnrnn.py`: LSTM(256) + mixture density head (5 gaussians per z dim),
  NLL loss, temperature-aware sampling.
- `s1_controller.py`: linear [z, h] -> 3 actions, tanh/sigmoid squashing.
- `cmaes.py`: minimal CMA-ES in numpy (Hansen's tutorial settings).
- `dream.py`: closed-loop latent rollouts, proxy reward, dream videos.
- `s1_data.py`: CarRacing-v3 collection at 64x64, cached under
  `data/stage1_ha_worldmodel/`.
- `train.py`: stages `vae`, `mdnrnn`, `controller`, or `all` (default).
- `tests/test_stage1_ha_worldmodel.py`: fast CPU checks.
- `exercises.md`: prediction exercises and break-it labs.

## Commands

    # everything, tiny, CPU, about 1-2 minutes
    uv run python -m stage1_ha_worldmodel.train --smoke

    # individual stages (smoke or full)
    uv run python -m stage1_ha_worldmodel.train vae --smoke
    uv run python -m stage1_ha_worldmodel.train mdnrnn --smoke
    uv run python -m stage1_ha_worldmodel.train controller --smoke
    uv run python -m stage1_ha_worldmodel.train controller --smoke --temperature 2.0
    uv run python -m stage1_ha_worldmodel.train controller --smoke --real

    # tests
    uv run pytest stage1_ha_worldmodel/tests -q

Outputs (checkpoints, `controller.npz`, `dream_vs_real_T*.gif`) go to
`data/stage1_ha_worldmodel/smoke/` or `.../full/`.

## Full run on one RTX 5090

Defaults in `train.py` are the full config; drop `--smoke`.

1. Collection: 2000 episodes x 1000 steps, about 2M frames. Box2D is
   CPU-bound; a single process takes 2-3 h, so shard across processes if
   impatient. Cached to one npz, reused afterwards.
2. `train.py vae --device cuda`: batch 256, 30k steps. About 1-1.5 h. Recon (sum MSE per
   frame) settling around 20-40 is normal.
3. `train.py mdnrnn --device cuda`: batch 128, seq len 64, 20k steps. About 1-1.5 h. NLL
   goes negative; that is expected for continuous densities.
4. `train.py controller`: CMA-ES pop 64, 400 generations in the dream at
   T = 1.15, minutes on CPU. For the faithful paper variant use `--real`
   (each candidate costs a real 1000-step episode, budget hours).

Expected outcome: dream returns climb steadily; the dream-trained controller
scores modestly in the real env because the proxy reward is not track reward
(see WHY.md, this is deliberate); `--real` training is what reaches the
600-900 real scores reported at worldmodels.github.io.

Modal cost: about $8-15 total (3-4 h of A100 at roughly $2.50/h for the two
model stages, plus a few CPU-hours for collection and CMA-ES).
