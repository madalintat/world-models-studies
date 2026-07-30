"""Train a ConvAE or ConvVAE on single CarRacing frames.

Smoke run (CPU, under two minutes, tiny data, what you actually execute here):
  uv run python -m stage0_compression.train --smoke
  uv run python -m stage0_compression.train --smoke --model ae
  uv run python -m stage0_compression.train --smoke --beta 0.0

Full run on one RTX 5090 (defaults below are this config):
  uv run python -m stage0_compression.collect --episodes 40 --frames-per-episode 500 --seed 0
  uv run python -m stage0_compression.train --model vae --device cuda
  20k frames, batch 128, 30k steps, Adam 1e-4. Expect 30 to 45 minutes wall
  clock. Outcome: recon loss settles around 20 to 40 (summed MSE per frame),
  KL around 30 to 60 nats, reconstructions show road shape and car position
  clearly but grass texture and curbs come out smoothed.
Modal cost: one A100/5090-class GPU for under an hour, roughly 2 to 4 USD.
"""

import argparse
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from stage0_compression import collect, viz
from stage0_compression.models import ConvAE, ConvVAE, ae_loss, vae_loss

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "stage0_compression"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["ae", "vae"], default="vae")
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (VAE only)")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--data", type=str, default=None, help="npz with a 'frames' array")
    p.add_argument("--out-dir", type=str, default=str(RUNS_DIR))
    p.add_argument("--tag", type=str, default=None, help="suffix for output file names")
    p.add_argument("--smoke", action="store_true", help="tiny data and steps, CPU")
    args = p.parse_args(argv)
    if args.smoke:
        args.steps = 150
        args.batch_size = 32
        args.lr = 1e-3
        args.device = "cpu"
    return args


def load_data(args):
    if args.data:
        return np.load(args.data)["frames"], args.data
    episodes, fpe = (2, 60) if args.smoke else (40, 500)
    path = collect.default_path(args.seed, episodes, fpe)
    frames = collect.ensure_dataset(path, episodes=episodes, frames_per_episode=fpe, seed=args.seed)
    return frames, str(path)


def train(args):
    if args.smoke:
        # A tiny model gains nothing from 20 threads, and capping them keeps
        # the smoke run fast even when the machine is busy.
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    frames, data_path = load_data(args)
    device = torch.device(args.device)

    model = (ConvVAE if args.model == "vae" else ConvAE)(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    log_every = max(1, args.steps // 10)
    t0 = time.time()

    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(frames), size=args.batch_size)
        x = viz.to_tensor(frames[idx]).to(device)
        if args.model == "vae":
            x_hat, mu, logvar, _ = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=args.beta)
        else:
            x_hat, _ = model(x)
            loss = recon = ae_loss(x, x_hat)
            kl = torch.tensor(0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            print(
                f"step {step:6d}  recon {recon.item():9.2f}  kl {kl.item():8.2f}  "
                f"total {loss.item():9.2f}  ({time.time() - t0:.0f}s)"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f"{args.model}_smoke" if args.smoke else args.model)
    config = {
        "model": args.model,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "data_path": data_path,
    }
    ckpt_path = out_dir / f"{tag}.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt_path)

    model.eval().to("cpu")
    sample_idx = rng.choice(len(frames), size=8, replace=False)
    grid = viz.recon_grid(model, frames[sample_idx])
    grid_path = out_dir / f"{tag}_recon.png"
    iio.imwrite(grid_path, grid)
    print(f"checkpoint: {ckpt_path}")
    print(f"recon grid: {grid_path}")
    return ckpt_path


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
