"""Collect CarRacing-v3 frames with a random policy and cache them as npz.

Frames are resized 96 -> 64, stored uint8 (N, 64, 64, 3). A small frame skip
makes consecutive stored frames visibly different, so the dataset is not
thousands of near-duplicates. Deterministic given the seed.

CLI:
  uv run python -m stage0_compression.collect --episodes 40 --frames-per-episode 500 --seed 0
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "stage0_compression"


def resize_frames(frames_u8):
    """(N, 96, 96, 3) uint8 -> (N, 64, 64, 3) uint8, area interpolation."""
    x = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0
    x = F.interpolate(x, size=(64, 64), mode="area")
    x = (x.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255)
    return x.to(torch.uint8).numpy()


def random_action(rng):
    # Mild steering plus steady gas keeps the car moving so frames vary.
    steer = float(np.clip(rng.normal(0.0, 0.4), -1.0, 1.0))
    gas = float(rng.uniform(0.2, 0.6))
    return np.array([steer, gas, 0.0], dtype=np.float32)


def collect_frames(episodes, frames_per_episode, seed=0, frame_skip=4, skip_start=30):
    env = gym.make("CarRacing-v3", render_mode=None)
    rng = np.random.default_rng(seed)
    all_frames = []
    for ep in tqdm(range(episodes), desc="episodes"):
        obs, _ = env.reset(seed=seed + ep)
        # The first frames are the zoom-in animation, useless as data.
        for _ in range(skip_start):
            obs, _, terminated, truncated, _ = env.step(random_action(rng))
        ep_frames = []
        done = False
        while len(ep_frames) < frames_per_episode and not done:
            action = random_action(rng)
            for _ in range(frame_skip):
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if done:
                    break
            ep_frames.append(obs.copy())
        all_frames.extend(ep_frames)
    env.close()
    return resize_frames(np.stack(all_frames))


def default_path(seed, episodes, frames_per_episode):
    return DATA_DIR / f"frames_e{episodes}_f{frames_per_episode}_seed{seed}.npz"


def ensure_dataset(path, episodes, frames_per_episode, seed=0, frame_skip=4):
    path = Path(path)
    if path.exists():
        return np.load(path)["frames"]
    frames = collect_frames(episodes, frames_per_episode, seed=seed, frame_skip=frame_skip)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=frames)
    print(f"saved {frames.shape[0]} frames to {path}")
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--frames-per-episode", type=int, default=500)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    out = (Path(args.out) if args.out
           else default_path(args.seed, args.episodes, args.frames_per_episode))
    frames = ensure_dataset(
        out, args.episodes, args.frames_per_episode, seed=args.seed, frame_skip=args.frame_skip
    )
    print(f"dataset: {frames.shape} {frames.dtype} at {out}")


if __name__ == "__main__":
    main()
