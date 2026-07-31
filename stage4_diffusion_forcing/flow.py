"""Flow matching core for diffusion forcing.

Conventions used everywhere in this stage:
  tau = 1.0 means clean data, tau = 0.0 means pure noise.
  The corruption path is linear: z_tau = (1 - tau) * noise + tau * x.
  The model is an x-predictor: given z_tau and tau it outputs an estimate
  of the clean x directly.
"""

import torch


def sample_frame_taus(
    batch: int,
    horizon: int,
    scheme: str = "uniform",
    mean: float = 0.0,
    std: float = 1.0,
    generator=None,
    device=None,
) -> torch.Tensor:
    """Diffusion forcing: every frame in every sequence gets its own
    independent tau. Shape (batch, horizon).

    scheme="uniform" draws tau uniformly on [0, 1], so every noise level is
    trained equally often. scheme="logit_normal" draws u ~ N(mean, std^2)
    and returns sigmoid(u), which is the SD3 recipe that nano-world-model
    uses by default. Sigmoid concentrates the mass near tau = 0.5: the
    middle noise levels, where the model actually has to decide what the
    frame contains. Near tau = 0 the answer is "anything" and near tau = 1
    it is "what you were already given", so both ends are cheap to fit and
    uniform sampling spends a lot of its budget there.
    """
    if scheme == "uniform":
        return torch.rand(batch, horizon, generator=generator, device=device)
    if scheme == "logit_normal":
        u = torch.randn(batch, horizon, generator=generator, device=device)
        return torch.sigmoid(u * std + mean)
    raise ValueError(f"unknown tau sampling scheme: {scheme}")


def interpolate(noise: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """z_tau = (1 - tau) * noise + tau * x.

    noise and x have shape (B, T, ...); tau has shape (B, T) and is
    broadcast over the trailing dims.
    """
    while tau.dim() < x.dim():
        tau = tau.unsqueeze(-1)
    return (1.0 - tau) * noise + tau * x


def loss_weight(tau: torch.Tensor, scheme: str = "ramp", eps: float = 1e-3) -> torch.Tensor:
    if scheme == "none":
        return torch.ones_like(tau)
    if scheme == "ramp":
        return 0.1 + 0.9 * tau
    if scheme == "v_space":
        # x-space MSE times 1/(1-tau)^2 equals the velocity-space MSE for
        # the linear path, so this recovers v-prediction's training signal.
        return 1.0 / torch.clamp(1.0 - tau, min=eps) ** 2
    raise ValueError(f"unknown weighting scheme: {scheme}")


def flow_matching_loss(
    pred_x: torch.Tensor,
    target_x: torch.Tensor,
    tau: torch.Tensor,
    weighting: str = "ramp",
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted MSE in x-space.

    pred_x, target_x: (B, T, S, D). tau: (B, T).
    frame_mask: optional (B, T) with 1 where the frame is supervised.
    Normalizing by the weight sum keeps the loss scale comparable across
    weighting schemes.
    """
    per_frame = ((pred_x - target_x) ** 2).mean(dim=(-2, -1))
    w = loss_weight(tau, weighting)
    if frame_mask is not None:
        w = w * frame_mask
    return (w * per_frame).sum() / w.sum().clamp(min=1e-8)


def make_ladder(k_steps: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Tau ladder for sampling: taus = [0, 1/K, ..., 1] (K+1 values) and the
    Euler mixing coefficients beta_s = (1 - tau_{s+1}) / (1 - tau_s), one per
    step. The last beta is 0, so the final latent is exactly the model's last
    x-prediction."""
    taus = torch.linspace(0.0, 1.0, k_steps + 1, device=device)
    betas = (1.0 - taus[1:]) / torch.clamp(1.0 - taus[:-1], min=1e-8)
    return taus, betas


def _full_sequence_schedule(k_steps: int, n_frames: int) -> torch.Tensor:
    """Every frame sits at the same noise level and they all walk up
    together. This is ordinary video diffusion: the sequence is one object.
    K + 1 rows."""
    rows = torch.arange(k_steps + 1).unsqueeze(1)
    return rows.expand(k_steps + 1, n_frames).clone()


def _sequential_schedule(k_steps: int, n_frames: int) -> torch.Tensor:
    """Frame f does not start until frame f-1 is completely clean. This is
    autoregression: K * n_frames + 1 rows, and the most expensive option.
    It is what `rollout` in sampling4.py does one frame at a time."""
    rows = torch.arange(k_steps * n_frames + 1).unsqueeze(1)
    offsets = torch.arange(n_frames).unsqueeze(0) * k_steps
    return (rows - offsets).clamp(0, k_steps)


def _pyramid_schedule(k_steps: int, n_frames: int, stagger: int = 1) -> torch.Tensor:
    """Frame f starts `stagger` rows after frame f-1, so the whole block is
    denoised in one sweeping wavefront.

    TODO(you): this is the interesting one, and it is about six lines.

    The contract, so a test can check you:
      - returns a LongTensor of shape (k_steps + stagger * (n_frames - 1) + 1,
        n_frames), holding ladder indices in [0, k_steps]
      - entry [r, f] is the ladder index of frame f at row r, which is
        (r - stagger * f), clamped into [0, k_steps]
      - row 0 is all zeros (everything is pure noise)
      - the last row is all k_steps (everything is clean)

    Worked example, k_steps=2, n_frames=3, stagger=1:

        row 0:  [0, 0, 0]     all noise
        row 1:  [1, 0, 0]     frame 0 has taken one ladder step
        row 2:  [2, 1, 0]     the wavefront moves right
        row 3:  [2, 2, 1]
        row 4:  [2, 2, 2]     all clean

    Why it matters, and why `stagger` is a real knob rather than a constant:
    a frame is denoised while the frames before it are still partly noisy.
    Small stagger means more parallelism (fewer rows, fewer model calls) but
    each frame is conditioned on blurrier context. Large stagger approaches
    `_sequential_schedule`, which is exactly the case stagger = k_steps.
    That tension is the whole reason diffusion forcing has a scheduling
    choice at all, and it is what break-it lab D asks you to measure.
    """
    raise NotImplementedError(
        "write _pyramid_schedule yourself: see the contract above and "
        "stage4_diffusion_forcing/exercises.md, break-it lab D"
    )


def scheduling_matrix(
    mode: str, k_steps: int, n_frames: int, stagger: int = 1
) -> torch.Tensor:
    """Ladder indices for every (row, frame) pair, shape (rows, n_frames).

    Diffusion forcing's real payoff is here rather than in training: because
    the model was trained with an independent tau per frame, it will accept
    any assignment of noise levels across a sequence at inference time. The
    schedule is therefore a free choice, and the three modes below are the
    corners of it. Row r tells you where every frame sits; the sampler walks
    the rows in order and Euler-mixes each frame from its current tau to its
    next one.
    """
    if mode == "full_sequence":
        return _full_sequence_schedule(k_steps, n_frames)
    if mode == "sequential":
        return _sequential_schedule(k_steps, n_frames)
    if mode == "pyramid":
        return _pyramid_schedule(k_steps, n_frames, stagger)
    raise ValueError(f"unknown scheduling mode: {mode}")


def euler_beta(tau_curr: torch.Tensor, tau_next: torch.Tensor, eps: float = 1e-8):
    """Mixing coefficient taking a latent from tau_curr to tau_next.

    Same algebra as `make_ladder`, but per element instead of per ladder
    step, so a scheduling matrix can move each frame by a different amount
    on the same row. beta = 1 leaves the latent untouched, which is what
    frames that do not move on this row need.
    """
    return (1.0 - tau_next) / torch.clamp(1.0 - tau_curr, min=eps)


def euler_mix(z: torch.Tensor, x_pred: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    """One ladder step. Algebraically identical to re-interpolating the
    implied noise with the predicted clean x at the next tau:
    z_next = (1 - tau_next) * eps_hat + tau_next * x_hat reduces to
    beta * z + (1 - beta) * x_hat."""
    return beta * z + (1.0 - beta) * x_pred


def noise_context(z_ctx: torch.Tensor, tau_ctx: float, generator=None) -> torch.Tensor:
    """Partially re-noise context latents to tau_ctx, matching the training
    distribution where context frames were independently corrupted."""
    noise = torch.randn(z_ctx.shape, generator=generator, device=z_ctx.device, dtype=z_ctx.dtype)
    return interpolate(noise, z_ctx, torch.as_tensor(tau_ctx, dtype=z_ctx.dtype, device=z_ctx.device))
