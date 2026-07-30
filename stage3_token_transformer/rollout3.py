"""Autoregressive rollout in token space, drift measurement, and video export.

Encode context frames to tokens, prefill the KV cache, then for each future
frame feed the (given) action token and sample 64 frame tokens one at a time.
Decode sampled tokens to pixels and compare against the ground-truth
continuation with PSNR per horizon step. Falling PSNR over the horizon is the
drift curve.
"""

import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from stage3_token_transformer.s3_data import frames_to_tensor, to_uint8_frames


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR between two uint8 images. Capped at 99 dB for identical inputs."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    return float(10.0 * np.log10(255.0**2 / mse))


def compute_drift(gen_frames: np.ndarray, gt_frames: np.ndarray) -> list[float]:
    """Per-horizon-step PSNR. Both inputs (N, 64, 64, 3) uint8."""
    assert gen_frames.shape == gt_frames.shape
    return [psnr(gen_frames[i], gt_frames[i]) for i in range(gen_frames.shape[0])]


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.no_grad()
def rollout(vqvae, gpt, context_frames, actions, n_future: int, temperature: float = 1.0):
    """context_frames: (T0, 64, 64, 3) uint8. actions: (T0 + n_future, 3)
    float32, actions[t] produced frame t. Returns (gen_frames uint8
    (n_future, 64, 64, 3), gen_tokens (n_future, 64))."""
    vqvae.eval()
    gpt.eval()
    T0 = context_frames.shape[0]
    K = gpt.cfg.tokens_per_frame
    device = next(gpt.parameters()).device

    ctx = frames_to_tensor(context_frames).to(device)
    ctx_tokens = vqvae.encode_to_indices(ctx)
    acts = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(device)

    emb = gpt.embed_sequence(acts[:T0].unsqueeze(0), ctx_tokens.unsqueeze(0))
    _, kv = gpt.step(emb)

    gen_tokens = []
    pending = None  # embedding of the last sampled token, not yet in the cache
    for f in range(n_future):
        a_emb = gpt.act_emb(acts[T0 + f]).view(1, 1, -1)
        x = a_emb if pending is None else torch.cat([pending, a_emb], dim=1)
        frame_toks = []
        for _ in range(K):
            logits, kv = gpt.step(x, kv)
            tok = _sample(logits[:, -1], temperature)
            frame_toks.append(tok)
            x = gpt.tok_emb(tok).unsqueeze(1)
        pending = x
        gen_tokens.append(torch.stack(frame_toks, dim=1))

    gen_tokens = torch.cat(gen_tokens, dim=0)
    pixels = vqvae.decode_from_indices(gen_tokens)
    return to_uint8_frames(pixels), gen_tokens.cpu().numpy()


def save_drift_csv(path, psnrs: list[float]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "psnr_db"])
        for i, p in enumerate(psnrs, start=1):
            w.writerow([i, f"{p:.3f}"])


def save_rollout_video(path, gen_frames: np.ndarray, gt_frames: np.ndarray, fps: int = 8):
    """Side-by-side (generated | ground truth) animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = np.full((64, 2, 3), 255, dtype=np.uint8)
    tiles = [np.concatenate([g, sep, t], axis=1) for g, t in zip(gen_frames, gt_frames)]
    imageio.mimsave(path, tiles, duration=1000.0 / fps, loop=0)
