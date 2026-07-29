"""End to end training for stage 4: latent AE, then diffusion forcing with
flow matching, then a tau-ladder rollout with a drift curve.

Smoke run (what you actually execute here, CPU, under two minutes):

    uv run python -m stage4_diffusion_forcing.train --smoke

FULL config, one RTX 5090 (documentation and defaults, not executed here):
  data:     100k frames (about 700 episodes of 150 kept frames), seed 0
  AE:       batch 256, 30k steps, Adam lr 3e-4, about 45 minutes
  dynamics: d_model 512, depth 12, heads 8, seq_len 32, batch 32 sequences,
            250k steps, Adam lr 3e-4 with cosine decay, weighting v_space
  wall clock: 8 to 11 hours total on the 5090
  expected outcome: 4-step ladder rollouts hold above roughly 20 dB PSNR past
  frame 50 with recognizable road geometry, where stage 3's teacher-forced
  transformer has usually fallen apart by frame 30 at equal compute.
  Modal: about 10 hours on one H100 at roughly $4/h, so on the order of $40.
  8 GPUs: plain DDP data parallel (one replica per GPU, batch 32 each,
  all-reduce gradients); the model is small enough that this was also all
  open-dreamer needed.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage4_diffusion_forcing.flow import (
    flow_matching_loss,
    interpolate,
    sample_frame_taus,
)
from stage4_diffusion_forcing.s4_latent_ae import LatentAE
from stage4_diffusion_forcing.s4_model import DynamicsTransformer
from stage4_diffusion_forcing.sampling4 import (
    drift_curve,
    rollout,
    save_drift_csv,
    save_rollout_video,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "stage4_diffusion_forcing"

SMOKE = dict(
    n_episodes=2,
    frames_per_episode=130,
    seed=0,
    # 1200 steps costs about 12 extra seconds and matters: at 150 steps the
    # decoder is still stuck outputting the dataset-mean image for every
    # latent, which makes the drift curve independent of the dynamics model
    ae_steps=1200,
    ae_batch=32,
    ae_base=16,
    dyn_steps=600,
    dyn_batch=8,
    seq_len=8,
    d_model=64,
    n_heads=4,
    depth=4,
    lr=3e-4,
    k_steps=4,
    prefill=4,
    horizon=12,
)

FULL = dict(
    n_episodes=700,
    frames_per_episode=150,
    seed=0,
    ae_steps=30_000,
    ae_batch=256,
    ae_base=32,
    dyn_steps=250_000,
    dyn_batch=32,
    seq_len=32,
    d_model=512,
    n_heads=8,
    depth=12,
    lr=3e-4,
    k_steps=4,
    prefill=8,
    horizon=56,
)


def resize64(obs: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = F.interpolate(t, size=(64, 64), mode="area")
    return (t[0].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()


def collect_data(cfg: dict, cache_name: str) -> dict:
    cache = DATA_DIR / cache_name
    if cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    import gymnasium as gym

    env = gym.make("CarRacing-v3", render_mode=None)
    rng = np.random.default_rng(cfg["seed"])
    frames, actions, ep_ids = [], [], []
    for ep in range(cfg["n_episodes"]):
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        # the first frames are the zoom-in animation, skip them
        for _ in range(30):
            obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
        steer = 0.0
        for _ in range(cfg["frames_per_episode"]):
            steer = float(np.clip(0.9 * steer + 0.25 * rng.normal(), -1.0, 1.0))
            gas = float(0.3 + 0.3 * rng.random())
            action = np.array([steer, gas, 0.0], dtype=np.float32)
            frames.append(resize64(obs))
            actions.append(action)
            ep_ids.append(ep)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    env.close()
    data = dict(
        frames=np.stack(frames),
        actions=np.stack(actions),
        ep_ids=np.array(ep_ids, dtype=np.int64),
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def train_ae(frames_u8: np.ndarray, cfg: dict, generator: torch.Generator) -> LatentAE:
    ae = LatentAE(latent_dim=8, base=cfg["ae_base"])
    opt = torch.optim.Adam(ae.parameters(), lr=cfg["lr"])
    frames = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0
    n = frames.shape[0]
    for step in range(cfg["ae_steps"]):
        idx = torch.randint(0, n, (cfg["ae_batch"],), generator=generator)
        batch = frames[idx]
        recon = ae(batch)
        loss = F.mse_loss(recon, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg["ae_steps"] // 4) == 0:
            print(f"  ae step {step:5d}  recon mse {loss.item():.5f}")
    return ae


@torch.no_grad()
def encode_all(ae: LatentAE, frames_u8: np.ndarray, batch: int = 128) -> torch.Tensor:
    ae.eval()
    frames = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0
    out = []
    for i in range(0, frames.shape[0], batch):
        z = ae.encode(frames[i : i + batch])
        out.append(LatentAE.to_tokens(z))
    ae.train()
    return torch.cat(out)  # (N, 64, 8)


def build_windows(ep_ids: np.ndarray, seq_len: int, stride: int = 2) -> np.ndarray:
    starts = []
    for ep in np.unique(ep_ids):
        idx = np.nonzero(ep_ids == ep)[0]
        for s in range(idx[0], idx[-1] - seq_len + 2, stride):
            starts.append(s)
    return np.array(starts, dtype=np.int64)


def train_dynamics(
    model: DynamicsTransformer,
    latents: torch.Tensor,
    actions: torch.Tensor,
    starts: np.ndarray,
    cfg: dict,
    weighting: str,
    clean_context: bool,
    generator: torch.Generator,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    t_len = cfg["seq_len"]
    for step in range(cfg["dyn_steps"]):
        pick = torch.randint(0, len(starts), (cfg["dyn_batch"],), generator=generator)
        s0 = torch.from_numpy(starts[pick.numpy()])
        idx = s0.unsqueeze(1) + torch.arange(t_len)
        z = latents[idx]
        a = actions[idx]
        b = z.shape[0]
        if clean_context:
            # teacher forcing baseline: context clean, only the last frame is
            # noised and supervised (this is the break-it lab, not the method)
            tau = torch.ones(b, t_len)
            tau[:, -1] = torch.rand(b, generator=generator)
            mask = torch.zeros(b, t_len)
            mask[:, -1] = 1.0
        else:
            tau = sample_frame_taus(b, t_len, generator=generator)
            mask = None
        noise = torch.randn(z.shape, generator=generator)
        z_tau = interpolate(noise, z, tau)
        pred = model(z_tau, a, tau)
        loss = flow_matching_loss(pred, z, tau, weighting=weighting, frame_mask=mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg["dyn_steps"] // 5) == 0:
            print(f"  dyn step {step:6d}  loss {loss.item():.5f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny CPU end to end run")
    p.add_argument("--weighting", default="ramp", choices=["none", "ramp", "v_space"])
    p.add_argument(
        "--clean-context",
        action="store_true",
        help="break-it lab: train with clean context (teacher forcing)",
    )
    p.add_argument(
        "--tau-ctx",
        type=float,
        default=None,
        help="context noise level at rollout (default 0.9, or 1.0 with --clean-context)",
    )
    p.add_argument("--k-steps", type=int, default=None, help="tau ladder steps")
    p.add_argument("--out", default=None, help="output directory")
    args = p.parse_args()

    cfg = dict(SMOKE if args.smoke else FULL)
    if args.k_steps is not None:
        cfg["k_steps"] = args.k_steps
    tau_ctx = args.tau_ctx
    if tau_ctx is None:
        tau_ctx = 1.0 if args.clean_context else 0.9

    if args.smoke:
        # small models gain nothing from many CPU threads, and this keeps the
        # smoke run predictable on a busy machine
        torch.set_num_threads(min(4, torch.get_num_threads()))

    tag = "smoke" if args.smoke else "full"
    out_dir = Path(args.out) if args.out else DATA_DIR / f"out_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    t_start = time.time()

    print("collecting data")
    data = collect_data(cfg, f"data_{tag}.npz")
    print(f"  {data['frames'].shape[0]} frames cached under {DATA_DIR}")

    print("training AE")
    ae = train_ae(data["frames"], cfg, generator)

    # guard against decoder collapse: an undertrained decoder can output the
    # dataset-mean image for every latent, and then the drift curve says
    # nothing about the dynamics model (see WHY.md, "Check reconstructions
    # first")
    with torch.no_grad():
        pair = data["frames"][[0, data["frames"].shape[0] // 2]]
        f = torch.from_numpy(pair).float().permute(0, 3, 1, 2) / 255.0
        two = ae.decode(ae.encode(f))
        gap = (two[0] - two[1]).abs().mean().item()
    if gap < 1e-3:
        print(
            "  WARNING: AE decoder outputs nearly identical images for "
            "different frames; train the AE longer or PSNR will not reflect "
            "the dynamics model"
        )

    print("encoding latents")
    latents = encode_all(ae, data["frames"])
    mean = latents.mean(dim=(0, 1))
    std = latents.std(dim=(0, 1)).clamp(min=1e-4)
    latents = (latents - mean) / std
    actions = torch.from_numpy(data["actions"])
    starts = build_windows(data["ep_ids"], cfg["seq_len"])

    print(f"training dynamics (weighting={args.weighting}, clean_context={args.clean_context})")
    model = DynamicsTransformer(
        latent_dim=8,
        n_latent_tokens=64,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        depth=cfg["depth"],
        max_t=cfg["seq_len"],
    )
    train_dynamics(
        model, latents, actions, starts, cfg, args.weighting, args.clean_context, generator
    )

    print(f"rollout: {cfg['k_steps']}-step ladder, tau_ctx={tau_ctx}")
    p0 = int(starts[0])
    prefill = latents[p0 : p0 + cfg["prefill"]].unsqueeze(0)
    span = cfg["prefill"] + cfg["horizon"]
    act_span = actions[p0 : p0 + span].unsqueeze(0)
    gen = rollout(
        model,
        prefill,
        act_span,
        horizon=cfg["horizon"],
        k_steps=cfg["k_steps"],
        tau_ctx=tau_ctx,
        generator=generator,
    )
    with torch.no_grad():
        z_grid = LatentAE.from_tokens(gen[0] * std + mean)
        pred_pixels = ae.decode(z_grid)
    gt_u8 = data["frames"][p0 + cfg["prefill"] : p0 + span]
    gt_pixels = torch.from_numpy(gt_u8).float().permute(0, 3, 1, 2) / 255.0
    curve = drift_curve(pred_pixels, gt_pixels)

    save_drift_csv(curve, str(out_dir / "drift.csv"))
    save_rollout_video(pred_pixels, gt_pixels, str(out_dir / "rollout.gif"))
    torch.save(
        dict(
            ae=ae.state_dict(),
            dynamics=model.state_dict(),
            latent_mean=mean,
            latent_std=std,
            cfg=cfg,
        ),
        out_dir / "checkpoint.pt",
    )
    print(f"drift PSNR: first {curve[0]:.2f} dB, last {curve[-1]:.2f} dB")
    print(f"wrote {out_dir}/drift.csv, rollout.gif, checkpoint.pt")
    print(f"done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
