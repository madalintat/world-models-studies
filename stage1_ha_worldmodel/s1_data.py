"""Rollout collection and caching for CarRacing-v3.

Frames are resized to 64x64 uint8. Actions come from a smoothed random
policy: a low-pass filter over piecewise-constant random targets, gas
biased positive so the car actually moves and the dataset contains motion.
Each stored step t pairs frames[t] (observation before acting) with
actions[t] (the action taken), so (z_t, a_t) -> z_{t+1} pairs fall out
directly.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

ZOOM_SKIP = 30  # CarRacing spends the first frames zooming in; skip them.


def resize64(frame: np.ndarray) -> np.ndarray:
    # np.array (not asarray) so the result is writable, which torch wants.
    return np.array(Image.fromarray(frame).resize((64, 64), Image.BILINEAR))


def collect(episodes: int, max_steps: int, seed: int, out_path: Path) -> dict:
    env = gym.make("CarRacing-v3")
    all_frames, all_actions, ep_lens = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        rng = np.random.default_rng(seed * 1000 + ep)
        for _ in range(ZOOM_SKIP):
            obs, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
        a = np.zeros(3)
        target = np.zeros(3)
        frames, actions = [], []
        for t in range(max_steps):
            if t % 10 == 0:
                brake = 0.0 if rng.random() < 0.9 else rng.uniform(0.0, 0.4)
                target = np.array([rng.uniform(-1, 1), rng.uniform(0.2, 1.0), brake])
            a = 0.8 * a + 0.2 * target
            act = a.astype(np.float32)
            frames.append(resize64(obs))
            actions.append(act)
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                break
        all_frames.append(np.stack(frames))
        all_actions.append(np.stack(actions))
        ep_lens.append(len(frames))
    env.close()
    data = dict(frames=np.concatenate(all_frames),
                actions=np.concatenate(all_actions),
                ep_lens=np.asarray(ep_lens, dtype=np.int64))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    return data


def load_or_collect(data_dir: Path, episodes: int, max_steps: int, seed: int) -> dict:
    path = data_dir / f"rollouts_e{episodes}_s{max_steps}_seed{seed}.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    print(f"collecting {episodes} episodes x {max_steps} steps (seed {seed}) ...")
    return collect(episodes, max_steps, seed, path)


def iter_episodes(data: dict):
    start = 0
    for n in data["ep_lens"]:
        yield data["frames"][start:start + n], data["actions"][start:start + n]
        start += n
