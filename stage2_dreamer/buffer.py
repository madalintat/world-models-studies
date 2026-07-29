"""Episode replay buffer for uint8 frames, sampling fixed-length subsequences."""

import numpy as np
import torch


class EpisodeBuffer:
    def __init__(self, capacity_steps=100_000, seq_len=16, seed=0):
        self.capacity_steps = capacity_steps
        self.seq_len = seq_len
        self.episodes = []
        self.rng = np.random.default_rng(seed)

    def add_episode(self, obs, action, reward, cont):
        """obs: (T, 64, 64, 3) uint8. action: (T, A) float32.
        reward, cont: (T,) float32. Index t holds the action that led into
        frame t and the reward received on arriving there (zeros at t=0)."""
        assert obs.dtype == np.uint8 and obs.shape[1:] == (64, 64, 3)
        T = obs.shape[0]
        assert action.shape[0] == T and reward.shape[0] == T
        if T < self.seq_len:
            return
        self.episodes.append({
            "obs": obs,
            "action": action.astype(np.float32),
            "reward": reward.astype(np.float32),
            "cont": cont.astype(np.float32),
        })
        while self.num_steps > self.capacity_steps and len(self.episodes) > 1:
            self.episodes.pop(0)

    @property
    def num_steps(self):
        return sum(ep["obs"].shape[0] for ep in self.episodes)

    def sample(self, batch_size, device="cpu"):
        assert self.episodes, "buffer is empty"
        lengths = np.array([ep["obs"].shape[0] for ep in self.episodes])
        probs = lengths / lengths.sum()
        out = {k: [] for k in ("obs", "action", "reward", "cont")}
        for _ in range(batch_size):
            ep = self.episodes[self.rng.choice(len(self.episodes), p=probs)]
            start = self.rng.integers(0, ep["obs"].shape[0] - self.seq_len + 1)
            for k in out:
                out[k].append(ep[k][start:start + self.seq_len])
        return {k: torch.as_tensor(np.stack(v)).to(device)
                for k, v in out.items()}
