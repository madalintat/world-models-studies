"""CarRacing-v3 frame collection for stage 3.

Convention: frames[t] is the observation, actions[t] is the action that was
applied just before frames[t] was observed. actions[0] is all zeros. This makes
the model sequence [a_t, z_t] read as "given the action, here is the frame it
produced".
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "stage3_token_transformer"


def resize_to_64(frames_u8: np.ndarray) -> np.ndarray:
    """(N, 96, 96, 3) uint8 -> (N, 64, 64, 3) uint8 via area interpolation."""
    t = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 255.0
    t = F.interpolate(t, size=(64, 64), mode="area")
    return (t.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()


def scripted_action(rng: np.random.Generator, t: int) -> np.ndarray:
    # Slow sinusoid on steering plus noise keeps the car on the road long
    # enough to see track, grass, and curbs without a trained policy.
    steer = 0.5 * np.sin(t / 25.0) + rng.normal(0.0, 0.1)
    gas = 0.3 + 0.3 * rng.random()
    brake = 0.05 if rng.random() < 0.05 else 0.0
    return np.array([np.clip(steer, -1, 1), gas, brake], dtype=np.float32)


def collect_episodes(n_episodes: int, steps: int, seed: int = 7, skip: int = 30):
    """Returns frames (E, steps+1, 64, 64, 3) uint8 and actions (E, steps+1, 3)
    float32, cached on disk keyed by the arguments."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"car_s{seed}_e{n_episodes}_n{steps}_k{skip}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["frames"], d["actions"]

    import gymnasium as gym

    env = gym.make("CarRacing-v3", render_mode=None)
    all_frames, all_actions = [], []
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed * 1000 + ep)
        obs, _ = env.reset(seed=seed * 1000 + ep)
        # The first frames are a zoom-in animation, not driving; skip them.
        for _ in range(skip):
            obs, *_ = env.step(np.array([0.0, 0.1, 0.0], dtype=np.float32))
        raw = [obs]
        acts = [np.zeros(3, dtype=np.float32)]
        for t in range(steps):
            a = scripted_action(rng, t)
            obs, _, terminated, truncated, _ = env.step(a)
            raw.append(obs)
            acts.append(a)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed * 1000 + ep + 500)
                for _ in range(skip):
                    obs, *_ = env.step(np.array([0.0, 0.1, 0.0], dtype=np.float32))
        all_frames.append(resize_to_64(np.stack(raw)))
        all_actions.append(np.stack(acts))
    env.close()

    frames = np.stack(all_frames)
    actions = np.stack(all_actions).astype(np.float32)
    np.savez_compressed(cache, frames=frames, actions=actions)
    return frames, actions
