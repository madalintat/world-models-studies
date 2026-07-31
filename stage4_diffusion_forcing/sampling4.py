"""Autoregressive rollout with the tau ladder and noised context, plus the
drift curve (PSNR vs ground truth over the horizon) for direct comparison
with stage 3."""

import numpy as np
import torch

from stage4_diffusion_forcing.flow import (
    euler_beta,
    euler_mix,
    make_ladder,
    noise_context,
    scheduling_matrix,
)


@torch.no_grad()
def rollout_block(
    model,
    prefill: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
    k_steps: int = 4,
    mode: str = "pyramid",
    stagger: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate `horizon` frames as one block under a scheduling matrix.

    Where `rollout` finishes each frame before starting the next, this
    denoises the whole block at once and lets the schedule decide how far
    ahead of frame f+1 frame f is allowed to get. The prefill stays clean
    at tau = 1 throughout, so no context re-noising is needed: unlike the
    autoregressive path, no frame here is ever conditioned on a finished
    generation of its own, only on partly-denoised neighbours.

    prefill: (B, T0, S, D) clean normalized latents.
    actions: (B, T0 + horizon, A). Requires T0 + horizon <= model.max_t.
    Returns the generated latents, (B, horizon, S, D).
    """
    model.eval()
    b, t0, s, d = prefill.shape
    total = t0 + horizon
    assert total <= model.max_t, (
        f"block rollout needs T0 + horizon ({total}) <= model.max_t "
        f"({model.max_t}); use rollout() for longer spans"
    )
    device = prefill.device
    taus, _ = make_ladder(k_steps, device=device)
    matrix = scheduling_matrix(mode, k_steps, horizon, stagger).to(device)

    z = torch.randn((b, horizon, s, d), generator=generator, device=device)
    ctx_tau = torch.ones(t0, device=device)
    for row in range(matrix.shape[0] - 1):
        curr, nxt = taus[matrix[row]], taus[matrix[row + 1]]
        tau_vec = torch.cat([ctx_tau, curr]).expand(b, -1)
        seq = torch.cat([prefill, z], dim=1)
        x_hat = model(seq, actions[:, :total], tau_vec)[:, t0:]
        beta = euler_beta(curr, nxt).view(1, horizon, 1, 1)
        z = euler_mix(z, x_hat, beta)
    return z


@torch.no_grad()
def rollout(
    model,
    prefill: torch.Tensor,
    actions: torch.Tensor,
    horizon: int,
    k_steps: int = 4,
    tau_ctx: float = 0.9,
    window: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate `horizon` new frames after the ground-truth prefill.

    prefill: (B, T0, S, D) clean normalized latents.
    actions: (B, T0 + horizon, A), recorded actions for the whole span.
    window: max sequence length fed to the model (defaults to model.max_t).
    Context frames the model generated itself are re-noised to tau_ctx each
    iteration; ground-truth prefill frames stay clean at tau = 1.
    Returns the generated latents, (B, horizon, S, D).
    """
    model.eval()
    b, t0, s, d = prefill.shape
    if window is None:
        window = model.max_t
    taus, betas = make_ladder(k_steps, device=prefill.device)
    frames = prefill.clone()
    is_gt = [True] * t0
    for t in range(t0, t0 + horizon):
        start = max(0, t + 1 - window)
        ctx = frames[:, start:t]
        flags = is_gt[start:t]
        if all(flags):
            ctx_in = ctx
        else:
            gt_mask = torch.tensor(flags, device=ctx.device).view(1, -1, 1, 1)
            ctx_in = torch.where(
                gt_mask, ctx, noise_context(ctx, tau_ctx, generator=generator)
            )
        ctx_tau = torch.tensor(
            [1.0 if f else tau_ctx for f in flags], device=ctx.device
        )
        z = torch.randn((b, 1, s, d), generator=generator, device=prefill.device)
        for step in range(k_steps):
            tau_vec = torch.cat([ctx_tau, taus[step : step + 1]]).expand(b, -1)
            seq = torch.cat([ctx_in, z], dim=1)
            x_hat = model(seq, actions[:, start : t + 1], tau_vec)[:, -1:]
            z = euler_mix(z, x_hat, betas[step])
        frames = torch.cat([frames, z], dim=1)
        is_gt.append(False)
    return frames[:, t0:]


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-frame PSNR in dB for pixel tensors in [0, 1], shape (T, ...)."""
    t = pred.shape[0]
    mse = ((pred - target) ** 2).reshape(t, -1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def drift_curve(pred_pixels: torch.Tensor, gt_pixels: torch.Tensor) -> np.ndarray:
    """pred_pixels, gt_pixels: (T, 3, 64, 64) in [0, 1]. Returns (T,) PSNR."""
    return psnr(pred_pixels, gt_pixels).cpu().numpy()


def save_drift_csv(curve: np.ndarray, path: str) -> None:
    with open(path, "w") as f:
        f.write("t,psnr_db\n")
        for i, v in enumerate(curve):
            f.write(f"{i},{v:.4f}\n")


def save_rollout_video(
    pred_pixels: torch.Tensor, gt_pixels: torch.Tensor, path: str, upscale: int = 3
) -> None:
    """Side by side gif, ground truth left, rollout right."""
    import imageio

    pred = (pred_pixels.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    gt = (gt_pixels.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    frames = []
    for p, g in zip(pred, gt):
        side = np.concatenate([g, p], axis=1)
        side = np.repeat(np.repeat(side, upscale, axis=0), upscale, axis=1)
        frames.append(side)
    imageio.mimsave(path, frames, duration=120, loop=0)
