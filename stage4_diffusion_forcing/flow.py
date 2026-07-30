"""Flow matching core for diffusion forcing.

Conventions used everywhere in this stage:
  tau = 1.0 means clean data, tau = 0.0 means pure noise.
  The corruption path is linear: z_tau = (1 - tau) * noise + tau * x.
  The model is an x-predictor: given z_tau and tau it outputs an estimate
  of the clean x directly.
"""

import torch


def sample_frame_taus(batch: int, horizon: int, generator=None, device=None) -> torch.Tensor:
    """Diffusion forcing: every frame in every sequence gets its own
    independent tau, uniform on [0, 1]. Shape (batch, horizon)."""
    return torch.rand(batch, horizon, generator=generator, device=device)


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
