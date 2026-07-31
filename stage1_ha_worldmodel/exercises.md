# Stage 1 exercises

Rules: commit to a written answer before running anything. The point is
calibrating your intuition, not being right. All commands run from the repo
root. Everything below uses the smoke configs so each run stays in the
couple-of-minutes range on a laptop CPU; keep the numbers qualitative.

Run this once first so checkpoints and data exist:

    uv run python -m stage1_ha_worldmodel.train --smoke

Checkpoints and videos land in `data/stage1_ha_worldmodel/smoke/`; the raw
rollout cache sits one level up in `data/stage1_ha_worldmodel/`, shared between
smoke and full runs.

## Prediction exercises

### P1: parameter budget

Write down, without running anything: how many parameters does the controller
have, and roughly how many does the MDN-RNN have (LSTM 35 -> 256 plus the head
256 -> 480)? Then check:

    uv run python -m stage1_ha_worldmodel.s1_controller
    uv run python -c "
    from stage1_ha_worldmodel.mdnrnn import MDNRNN
    print(sum(p.numel() for p in MDNRNN().parameters()))"

Expected: controller 867. MDN-RNN about 423k, so roughly 500x the controller.
Sit with that ratio: nearly all learned capacity is in the model of the world,
almost none in the policy. That imbalance is the paper's thesis.

### P2: NLL is not bounded below by zero

Prediction: after the smoke MDN-RNN training, is the NLL positive or negative?
Commit, then rerun and read the `[mdnrnn]` lines:

    uv run python -m stage1_ha_worldmodel.train mdnrnn --smoke

Expected: it can go negative, because this is a continuous density (a gaussian
with std 0.1 has log density about 1.4 at its mean, so around -1 per dimension;
the printed NLL sums 32 dimensions, so it can sink well below -20). If you
predicted "loss must be positive" you were thinking of cross-entropy over
discrete classes.

### P3: where does the dream video fall apart

The smoke video `dream_vs_real_T1.0.gif` shows real | VAE reconstruction |
dream, with the dream fed the same actions as the real episode. Prediction: for
how many frames does the dream panel stay recognizably in sync with the real
panel, and what dies first: the road shape, the car's position on the road, or
overall color? Watch it, count.

Expected with smoke-sized training: sync for only a handful of frames, then the
dream drifts to a plausible-looking but different road, and eventually to
smear. The middle panel shows how much of the mush is the VAE's fault versus
the dynamics model's fault. A fully trained M holds together for hundreds of
steps.

### P4: dream return vs generations

Prediction: over the 3 smoke CMA-ES generations, does the best dream return
improve monotonically? Commit, then:

    uv run python -m stage1_ha_worldmodel.train controller --smoke

Expected: generally upward but not necessarily monotone; with population 8 and
one rollout per candidate, evaluation noise is large. Note the mechanism:
CMA-ES only needs a ranking, and a noisy ranking still carries signal. Full
runs average several rollouts per candidate for exactly this reason.

### P5: what the proxy converges to

Read `proxy_reward` in `dream.py`. Prediction: after enough CMA-ES generations
on this proxy, what does the controller's gas output look like, and is that a
bug? Verify by printing actions of the best controller:

    uv run python -c "
    import numpy as np, torch
    from stage1_ha_worldmodel.s1_controller import Controller
    d = np.load('data/stage1_ha_worldmodel/smoke/controller.npz')
    c = Controller(); c.set_flat_params(d['params'])
    z, h = torch.randn(8, 32), torch.zeros(8, 256)
    print(c(z, h))"

Expected: gas pushed toward 1, brake toward 0 (3 smoke generations may only get
partway). Not a bug: the proxy pays for throttle, so throttle is what you get.
This is objective misspecification in its most legible form.

## Break-it labs

### B1: temperature sweep, determinism vs chaos

Generate three dream videos from the same trained smoke checkpoints:

    uv run python -m stage1_ha_worldmodel.train controller --smoke --temperature 0.1
    uv run python -m stage1_ha_worldmodel.train controller --smoke --temperature 1.0
    uv run python -m stage1_ha_worldmodel.train controller --smoke --temperature 2.0

Compare `dream_vs_real_T0.1.gif`, `T1.0`, `T2.0`. What to observe: at 0.1 the
dream is smooth, slow-changing, almost frozen; the model keeps picking the
single most likely branch with tiny noise, and errors compound gently in one
direction. At 1.0 you see the model's honest uncertainty. At 2.0 the mixture
picks improbable components often and stds are inflated, so the dream jitters
and falls off-manifold fast. What it teaches: temperature is a dial between
"exploitable fiction" and "unlearnable noise", and usable dreams live in
between.

### B2: the cheating lab

Train the controller in an ice-cold dream, then grade it in reality:

    uv run python -m stage1_ha_worldmodel.train controller --smoke --temperature 0.1
    uv run python -c "
    import numpy as np, torch
    from stage1_ha_worldmodel.train import real_return, DATA_DIR
    from stage1_ha_worldmodel.s1_vae import ConvVAE
    from stage1_ha_worldmodel.mdnrnn import MDNRNN
    from stage1_ha_worldmodel.s1_controller import Controller
    ck = DATA_DIR / 'smoke'
    vae = ConvVAE()
    vae.load_state_dict(torch.load(ck / 'vae.pt', weights_only=True))
    vae.eval()
    m = MDNRNN()
    m.load_state_dict(torch.load(ck / 'mdnrnn.pt', weights_only=True))
    m.eval()
    c = Controller()
    c.set_flat_params(np.load(ck / 'controller.npz')['params'])
    print('real return:', real_return(c, vae, m, seed=123, max_steps=200))"

What to observe: the dream returns printed during training look healthy and
climb; the real return is poor (often negative once the car leaves the road,
since CarRacing charges -0.1 per frame and pays only for new tiles). What it
teaches: a controller optimized in a too-deterministic dream, on a proxy
objective, learns levers that exist only in the dream. This gap is the central
obstacle of the entire field. Repeat the real evaluation after training at
temperature 1.0 and the gap narrows slightly; nothing about the proxy is fixed,
but the policy is at least robust to model noise.

### B3: single gaussian, mushy futures

Sabotage: in `train.py`, construct the MDN-RNN with `n_gaussians=1` (edit the
`MDNRNN()` call in `train_mdnrnn`, and the ones in `main`), then rerun:

    uv run python -m stage1_ha_worldmodel.train --smoke

What to observe: NLL still decreases, often to a similar-looking number, so the
loss curve alone does not expose the problem. The dream video does: with one
gaussian the sampled futures hug a single averaged trajectory, corners come out
as indecisive gray blends, and long rollouts look smoother but wronger. What it
teaches: a healthy loss curve can hide a model that has averaged away exactly
the structure (multimodality) you built the MDN for. Revert the edit
afterwards.
