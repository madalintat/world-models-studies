"""Stage 3 training: VQ-VAE tokenizer, then autoregressive dynamics.

Run:
    uv run python -m stage3_token_transformer.train --smoke
    uv run python -m stage3_token_transformer.train --smoke --phase vq
    uv run python -m stage3_token_transformer.train --smoke --phase dyn

FULL config (defaults below, documented for one RTX 5090, 32 GB):
  Data: 250 episodes x 200 steps = 50k frames (seed 7), ~40 min to collect once.
  Phase vq: base 128, batch 128, 60k steps, lr 3e-4. About 1h wall clock.
    Expected: recon MSE below 1e-3, codebook perplexity above 100 with all the
    stabilizers on, reconstructions visually near-perfect except tiny text.
  Phase dyn: d_model 512, 8 layers, 8 heads, 20-frame windows (1300 tokens),
    batch 12, 100k steps, lr 3e-4. About 3-6h wall clock.
    Expected: next-token accuracy 60-80%, rollouts hold the road geometry for
    20+ frames before drifting, drift PSNR bends down near the context length.
  Modal cost: roughly $12-25 total (5-7 GPU hours on an A100/H100 class card).

--smoke shrinks everything (2 short episodes, tiny models, ~2 min on CPU),
trains both phases, and writes a 5-frame rollout GIF plus drift.csv.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from stage3_token_transformer.rollout3 import (
    compute_drift,
    rollout,
    save_drift_csv,
    save_rollout_video,
)
from stage3_token_transformer.s3_data import collect_episodes
from stage3_token_transformer.s3_transformer import (
    IGNORE_INDEX,
    GPTConfig,
    TokenGPT,
    make_targets,
    sequence_loss,
)
from stage3_token_transformer.vqvae import VQVAE

FULL = dict(
    episodes=250,
    steps=200,
    seed=7,
    vq_base=128,
    vq_batch=128,
    vq_steps=60_000,
    vq_decay=0.99,
    d_model=512,
    n_layers=8,
    n_heads=8,
    seq_frames=20,
    max_frames=24,
    dyn_batch=12,
    dyn_steps=100_000,
    lr=3e-4,
    context=8,
    future=16,
    log_every=500,
)

SMOKE = dict(
    episodes=2,
    steps=130,
    seed=7,
    vq_base=32,
    vq_batch=32,
    vq_steps=150,
    # Faster EMA decay so codebook dynamics are visible within 150 steps.
    vq_decay=0.95,
    d_model=64,
    n_layers=2,
    n_heads=2,
    seq_frames=6,
    max_frames=10,
    dyn_batch=8,
    dyn_steps=120,
    lr=3e-4,
    context=4,
    future=5,
    log_every=30,
)


def out_dir_for(smoke: bool) -> Path:
    root = Path(__file__).resolve().parents[1] / "data" / "stage3_token_transformer" / "runs"
    return root / ("smoke" if smoke else "full")


def frames_to_tensor(frames_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0


def train_vq(cfg, frames, out_dir, ema=True, dead_reinit=True):
    flat = frames.reshape(-1, 64, 64, 3)
    model = VQVAE(
        num_codes=256,
        code_dim=64,
        base=cfg["vq_base"],
        decay=cfg["vq_decay"],
        enable_ema=ema,
        enable_dead_reinit=dead_reinit,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(0)
    model.train()
    for step in range(cfg["vq_steps"]):
        idx = rng.integers(0, flat.shape[0], size=cfg["vq_batch"])
        x = frames_to_tensor(flat[idx])
        loss, logs = model.loss(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg["log_every"] == 0 or step == cfg["vq_steps"] - 1:
            print(
                f"[vq] step {step:6d} recon {logs['recon']:.4f} "
                f"commit {logs['commit']:.4f} perplexity {logs['perplexity']:.1f} "
                f"active {model.quantizer.active_codes()}/256"
            )
    hist = model.quantizer.usage_histogram(normalize=True).numpy()
    np.savetxt(out_dir / "codebook_usage.csv", hist, header="usage_fraction", comments="")
    torch.save({"state_dict": model.state_dict(), "base": cfg["vq_base"]}, out_dir / "vqvae.pt")
    return model


def load_vq(out_dir) -> VQVAE:
    ckpt = torch.load(out_dir / "vqvae.pt", weights_only=True)
    model = VQVAE(num_codes=256, code_dim=64, base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def encode_dataset(vqvae, frames) -> np.ndarray:
    """frames (E, T, 64, 64, 3) uint8 -> tokens (E, T, 64) int64."""
    vqvae.eval()
    E, T = frames.shape[:2]
    out = np.empty((E, T, 64), dtype=np.int64)
    for e in range(E):
        for start in range(0, T, 64):
            chunk = frames_to_tensor(frames[e, start : start + 64])
            out[e, start : start + 64] = vqvae.encode_to_indices(chunk).numpy()
    return out


def train_dyn(cfg, tokens, actions, out_dir):
    gpt_cfg = GPTConfig(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_frames=cfg["max_frames"],
    )
    model = TokenGPT(gpt_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(1)
    E, T = tokens.shape[:2]
    W = cfg["seq_frames"]
    model.train()
    for step in range(cfg["dyn_steps"]):
        eps = rng.integers(0, E, size=cfg["dyn_batch"])
        starts = rng.integers(0, T - W, size=cfg["dyn_batch"])
        tok = torch.from_numpy(np.stack([tokens[e, s : s + W] for e, s in zip(eps, starts)]))
        act = torch.from_numpy(np.stack([actions[e, s : s + W] for e, s in zip(eps, starts)]))
        logits = model(act, tok)
        loss = sequence_loss(logits, tok)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg["log_every"] == 0 or step == cfg["dyn_steps"] - 1:
            with torch.no_grad():
                tgt = make_targets(tok)
                valid = tgt != IGNORE_INDEX
                acc = (logits.argmax(-1)[valid] == tgt[valid]).float().mean().item()
            print(f"[dyn] step {step:6d} ce {loss.item():.4f} acc {acc:.3f}")
    torch.save({"state_dict": model.state_dict(), "cfg": vars(gpt_cfg)}, out_dir / "gpt.pt")
    return model


def load_dyn(out_dir) -> TokenGPT:
    ckpt = torch.load(out_dir / "gpt.pt", weights_only=True)
    model = TokenGPT(GPTConfig(**ckpt["cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    return model


def run_rollout(cfg, vqvae, gpt, frames, actions, out_dir, temperature):
    T0, N = cfg["context"], cfg["future"]
    start = 40  # well past the zoom skip, mid-drive
    ep = 0
    ctx = frames[ep, start : start + T0]
    gt = frames[ep, start + T0 : start + T0 + N]
    acts = actions[ep, start : start + T0 + N]
    gen, _ = rollout(vqvae, gpt, ctx, acts, n_future=N, temperature=temperature)
    psnrs = compute_drift(gen, gt)
    save_drift_csv(out_dir / "drift.csv", psnrs)
    save_rollout_video(out_dir / "rollout.gif", gen, gt)
    print("[rollout] PSNR per horizon step:", [f"{p:.2f}" for p in psnrs])
    print(f"[rollout] wrote {out_dir / 'drift.csv'} and {out_dir / 'rollout.gif'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["vq", "dyn", "all"], default="all")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--context", type=int, default=None, help="context frames for the rollout")
    p.add_argument("--future", type=int, default=None, help="future frames for the rollout")
    p.add_argument("--no-ema", action="store_true", help="break-it lab: freeze codebook updates")
    p.add_argument("--no-dead-reinit", action="store_true", help="break-it lab: no dead-code reinit")
    args = p.parse_args()

    if args.smoke:
        # Small tensors thrash on many-core CPUs when torch grabs every thread.
        torch.set_num_threads(4)

    cfg = dict(SMOKE if args.smoke else FULL)
    # Seed torch too (numpy generators are seeded locally): weight init and
    # temperature sampling both draw from the global torch RNG, and repeated
    # runs should be reproducible, e.g. the temperature-0 lab in exercises.md.
    torch.manual_seed(0)
    if args.context is not None:
        cfg["context"] = args.context
    if args.future is not None:
        cfg["future"] = args.future
    if cfg["context"] + cfg["future"] > cfg["max_frames"]:
        cfg["max_frames"] = cfg["context"] + cfg["future"]

    out_dir = out_dir_for(args.smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    frames, actions = collect_episodes(cfg["episodes"], cfg["steps"], seed=cfg["seed"])
    print(f"[data] frames {frames.shape} actions {actions.shape} ({time.time() - t0:.1f}s)")

    vqvae = None
    if args.phase in ("vq", "all"):
        vqvae = train_vq(cfg, frames, out_dir, ema=not args.no_ema,
                         dead_reinit=not args.no_dead_reinit)
    if args.phase in ("dyn", "all"):
        if vqvae is None:
            if not (out_dir / "vqvae.pt").exists():
                raise SystemExit("no vqvae.pt found, run --phase vq (or all) first")
            vqvae = load_vq(out_dir)
        tokens = encode_dataset(vqvae, frames)
        gpt = train_dyn(cfg, tokens, actions, out_dir)
        run_rollout(cfg, vqvae, gpt, frames, actions, out_dir, args.temperature)

    print(f"[done] total {time.time() - t0:.1f}s, outputs in {out_dir}")


if __name__ == "__main__":
    main()
