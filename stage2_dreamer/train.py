"""Dreamer training loop: collect, train world model, train actor-critic in
imagination, repeat.

Smoke run (CPU, under two minutes):
    uv run python -m stage2_dreamer.train --smoke

Full run, one RTX 5090 (documented, defaults below in full_config):
    uv run python -m stage2_dreamer.train
    500k env steps at action_repeat 2, batch 32 x seq 16, roughly 50k world
    model and 50k actor-critic gradient steps. Expect 6 to 10 hours wall
    clock. Outcome: the car visibly follows the road; episode returns of
    600 to 900 are typical, against roughly minus 50 for a random policy.
Modal cost, one line: about 8 h on an A100 40GB at roughly $2.5/h, so $20
per full run, give or take a factor of two for the exact GPU and duration.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from stage2_dreamer.buffer import EpisodeBuffer
from stage2_dreamer.s2_ac import HORIZON, Actor, Critic, ac_losses
from stage2_dreamer.s2_wm import WorldModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "stage2_dreamer"


@dataclass
class Config:
    seed: int = 0
    total_env_steps: int = 500_000
    episode_max_steps: int = 1000
    action_repeat: int = 2
    buffer_capacity: int = 200_000
    batch_size: int = 32
    seq_len: int = 16
    wm_steps_per_iter: int = 100
    ac_steps_per_iter: int = 100
    wm_lr: float = 3e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    grad_clip: float = 100.0
    horizon: int = HORIZON
    kl_alpha: float | None = None  # None uses KL_BALANCE_ALPHA from rssm.py
    det_z: bool = False
    depth: int = 32
    embed_dim: int = 1024
    deter_dim: int = 512
    hidden_dim: int = 512
    device: str = "cpu"
    smoke: bool = False


def smoke_config():
    return Config(
        total_env_steps=100, episode_max_steps=100, action_repeat=1,
        batch_size=4, wm_steps_per_iter=3, ac_steps_per_iter=3,
        depth=8, embed_dim=128, deter_dim=64, hidden_dim=64, smoke=True)


def resize64(frame):
    """(96, 96, 3) uint8 to (64, 64, 3) uint8."""
    x = torch.as_tensor(frame).movedim(-1, 0).unsqueeze(0).float()
    x = F.interpolate(x, size=(64, 64), mode="area")
    return x.squeeze(0).movedim(0, -1).round().clamp(0, 255).byte().numpy()


def to_env_action(a):
    """Map a tanh-space action in [-1, 1]^3 onto CarRacing's action box."""
    return np.array([a[0], (a[1] + 1) / 2, (a[2] + 1) / 2], dtype=np.float32)


class Agent:
    """Runs the actor in the real env, carrying the recurrent state."""

    def __init__(self, wm, actor, device):
        self.wm, self.actor, self.device = wm, actor, device
        self.reset()

    def reset(self):
        self.h, self.z = self.wm.rssm.initial(1, self.device)
        self.prev_action = torch.zeros(1, 3, device=self.device)

    @torch.no_grad()
    def __call__(self, obs64):
        x = self.wm.preprocess(
            torch.as_tensor(obs64, device=self.device)).unsqueeze(0)
        embed = self.wm.encoder(x)
        self.h, self.z, _, _ = self.wm.rssm.obs_step(
            self.h, self.z, self.prev_action, embed)
        action, _ = self.actor.sample(torch.cat([self.h, self.z], -1))
        self.prev_action = action
        return action.squeeze(0).cpu().numpy()


def collect_episode(env, policy, cfg, seed=None):
    """Returns arrays aligned so index t holds (obs_t, action into t,
    reward on arrival at t, cont at t)."""
    obs, _ = env.reset(seed=seed)
    obs = resize64(obs)
    frames, actions = [obs], [np.zeros(3, np.float32)]
    rewards, conts = [0.0], [1.0]
    for _ in range(cfg.episode_max_steps):
        a = policy(obs).astype(np.float32)
        reward, done = 0.0, False
        for _ in range(cfg.action_repeat):
            nxt, r, term, trunc, _ = env.step(to_env_action(a))
            reward += float(r)
            done = term or trunc
            if done:
                break
        obs = resize64(nxt)
        frames.append(obs)
        actions.append(a)
        rewards.append(reward)
        conts.append(0.0 if term else 1.0)
        if done:
            break
    return (np.stack(frames), np.stack(actions),
            np.array(rewards, np.float32), np.array(conts, np.float32))


def collect_random_episode_cached(env, cfg):
    """First episode uses a random policy with a fixed seed; cache it on disk
    so smoke reruns skip the env rollout."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / (f"prefill_seed{cfg.seed}_len{cfg.episode_max_steps}"
                        f"_rep{cfg.action_repeat}.npz")
    if cache.exists():
        d = np.load(cache)
        return d["obs"], d["action"], d["reward"], d["cont"]
    rng = np.random.default_rng(cfg.seed)

    def random_policy(_obs):
        return rng.uniform(-1, 1, size=3)

    obs, action, reward, cont = collect_episode(
        env, random_policy, cfg, seed=cfg.seed)
    np.savez_compressed(cache, obs=obs, action=action,
                        reward=reward, cont=cont)
    return obs, action, reward, cont


def train(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cpu":
        # At these tensor sizes torch's default thread pool is roughly 20x
        # slower than a single thread on ops like small convs and interpolate.
        torch.set_num_threads(1)
    env = gym.make("CarRacing-v3")

    wm = WorldModel(action_dim=3, depth=cfg.depth, embed_dim=cfg.embed_dim,
                    deter_dim=cfg.deter_dim,
                    hidden_dim=cfg.hidden_dim).to(device)
    wm.rssm.deterministic_z = cfg.det_z
    actor = Actor(wm.feat_dim, 3, cfg.hidden_dim).to(device)
    critic = Critic(wm.feat_dim, cfg.hidden_dim).to(device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.wm_lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    buffer = EpisodeBuffer(cfg.buffer_capacity, cfg.seq_len, cfg.seed)
    agent = Agent(wm, actor, device)

    env_steps, iteration = 0, 0
    t0 = time.time()
    last_metrics = {}
    while env_steps < cfg.total_env_steps:
        if iteration == 0:
            episode = collect_random_episode_cached(env, cfg)
        else:
            agent.reset()
            episode = collect_episode(env, agent, cfg)
        buffer.add_episode(*episode)
        env_steps += (episode[0].shape[0] - 1) * cfg.action_repeat
        ep_return = float(episode[2].sum())

        start_states = None
        for _ in range(cfg.wm_steps_per_iter):
            batch = buffer.sample(cfg.batch_size, device)
            wm_opt.zero_grad()
            total, wm_metrics, states = wm.loss(batch, kl_alpha=cfg.kl_alpha)
            total.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
            wm_opt.step()
            start_states = states
            last_metrics.update(wm_metrics)

        for _ in range(cfg.ac_steps_per_iter):
            h0 = start_states[0].flatten(0, 1)
            z0 = start_states[1].flatten(0, 1)
            actor_loss, critic_loss, ac_metrics = ac_losses(
                wm, actor, critic, h0, z0, cfg.horizon)
            # The actor only needs gradients through activations; freezing
            # the world model and critic skips their weight-gradient work.
            wm.requires_grad_(False)
            critic.requires_grad_(False)
            actor_opt.zero_grad()
            actor_loss.backward()
            wm.requires_grad_(True)
            critic.requires_grad_(True)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
            actor_opt.step()
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
            critic_opt.step()
            last_metrics.update(ac_metrics)

        iteration += 1
        line = " ".join(f"{k}={v:.3f}" for k, v in sorted(last_metrics.items()))
        print(f"iter {iteration} env_steps {env_steps} "
              f"return {ep_return:.1f} {line}", flush=True)

        if cfg.smoke:
            bad = [k for k, v in last_metrics.items() if not np.isfinite(v)]
            assert not bad, f"non-finite losses: {bad}"
            print(f"SMOKE OK in {time.time() - t0:.1f}s: all losses finite")
            break
    env.close()
    return last_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run: 1 short episode, a few gradient steps")
    p.add_argument("--kl-alpha", type=float, default=None,
                   help="override KL balance alpha (break-it lab: try 0.0)")
    p.add_argument("--det-z", action="store_true",
                   help="argmax instead of sampling z (break-it lab)")
    p.add_argument("--horizon", type=int, default=None,
                   help="imagination horizon (prediction exercise: try 50)")
    p.add_argument("--device", default=None,
                   help="training device for the full run, e.g. cuda")
    args = p.parse_args()
    cfg = smoke_config() if args.smoke else Config()
    if args.device is not None:
        cfg.device = args.device
    if args.kl_alpha is not None:
        cfg.kl_alpha = args.kl_alpha
    if args.det_z:
        cfg.det_z = True
    if args.horizon is not None:
        cfg.horizon = args.horizon
    train(cfg)


if __name__ == "__main__":
    main()
