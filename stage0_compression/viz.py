"""Visualizations: reconstruction grid, latent traversal, interpolation strip.

All functions take uint8 HWC frames in and return uint8 HWC images out, so
the outputs can be written directly with imageio.

CLI (after training saved a checkpoint):
  uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_smoke.pt --mode grid
  uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_smoke.pt --mode traversal --dims 0 1 2 3
  uv run python -m stage0_compression.viz --checkpoint runs/stage0_compression/vae_smoke.pt --mode interp
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from stage0_compression.models import ConvAE, ConvVAE


def _to_tensor(frames_u8):
    """(N, 64, 64, 3) uint8 -> (N, 3, 64, 64) float in [0, 1]."""
    return torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0


def _to_uint8(x):
    """(N, 3, 64, 64) float -> (N, 64, 64, 3) uint8."""
    x = (x.detach().clamp(0, 1) * 255.0).round()
    return x.permute(0, 2, 3, 1).to(torch.uint8).numpy()


@torch.no_grad()
def encode_deterministic(model, x):
    """One code per frame for viz: the AE latent, or the VAE posterior mean."""
    if isinstance(model, ConvVAE):
        mu, _ = model.encode(x)
        return mu
    return model.encode(x)


@torch.no_grad()
def recon_grid(model, frames_u8):
    """Originals on the top row, reconstructions on the bottom row."""
    x = _to_tensor(frames_u8)
    x_hat = model.decode(encode_deterministic(model, x))
    top = np.concatenate(list(frames_u8), axis=1)
    bottom = np.concatenate(list(_to_uint8(x_hat)), axis=1)
    return np.concatenate([top, bottom], axis=0)


@torch.no_grad()
def latent_traversal(model, frame_u8, dims, span=3.0, steps=7):
    """One row per dim: decode z with that dim swept from -span to +span.
    Returns (len(dims)*64, steps*64, 3) uint8."""
    z0 = encode_deterministic(model, _to_tensor(frame_u8[None]))
    rows = []
    for d in dims:
        codes = z0.repeat(steps, 1)
        codes[:, d] = torch.linspace(-span, span, steps)
        imgs = _to_uint8(model.decode(codes))
        rows.append(np.concatenate(list(imgs), axis=1))
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def interpolation_strip(model, frame_a_u8, frame_b_u8, steps=8):
    """Decode evenly spaced points on the line between two codes.
    Returns (64, steps*64, 3) uint8."""
    za = encode_deterministic(model, _to_tensor(frame_a_u8[None]))
    zb = encode_deterministic(model, _to_tensor(frame_b_u8[None]))
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    imgs = _to_uint8(model.decode(za * (1 - t) + zb * t))
    return np.concatenate(list(imgs), axis=1)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    cls = ConvVAE if cfg["model"] == "vae" else ConvAE
    model = cls(latent_dim=cfg["latent_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--mode", choices=["grid", "traversal", "interp"], default="grid")
    p.add_argument("--dims", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--span", type=float, default=3.0)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    model, cfg = load_checkpoint(args.checkpoint)
    data_path = args.data or cfg["data_path"]
    frames = np.load(data_path)["frames"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(frames), size=max(args.n, 2), replace=False)

    if args.mode == "grid":
        img = recon_grid(model, frames[idx[: args.n]])
    elif args.mode == "traversal":
        img = latent_traversal(model, frames[idx[0]], args.dims, span=args.span)
    else:
        img = interpolation_strip(model, frames[idx[0]], frames[idx[1]], steps=args.n)

    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(
        f"{Path(args.checkpoint).stem}_{args.mode}.png"
    )
    iio.imwrite(out, img)
    print(f"wrote {out} shape={img.shape}")


if __name__ == "__main__":
    main()
