"""Actor and critic trained entirely inside the world model's imagination."""

import torch
import torch.distributions as td
import torch.nn as nn

from stage2_dreamer.s2_wm import symexp, symlog

GAMMA = 0.997
LAMBDA = 0.95
HORIZON = 15
ENTROPY_BETA = 3e-4
MIN_STD = 0.1


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy on [h, z]. Actions live in [-1, 1]^3."""

    def __init__(self, feat_dim, action_dim=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)

    def forward(self, feat):
        x = self.net(feat)
        return self.mean(x), nn.functional.softplus(self.std(x)) + MIN_STD

    def sample(self, feat):
        """Reparameterized action plus entropy of the pre-tanh Gaussian.

        The tanh correction to the entropy is dropped: as a bonus term we only
        need a pressure toward wider distributions, not the exact value.
        """
        mean, std = self(feat)
        action = torch.tanh(mean + std * torch.randn_like(std))
        entropy = td.Normal(mean, std).entropy().sum(-1)
        return action, entropy


class Critic(nn.Module):
    """Predicts the symlog of the lambda return of [h, z]."""

    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


def imagine(wm, actor, h, z, horizon):
    """Roll the prior forward with the actor choosing actions.

    Returns feats (H+1, N, feat_dim) including the start state, and the
    per-step action entropies (H, N). Gradients flow through actions into
    future states: that path is the whole point of backprop through dream.
    """
    feats = [torch.cat([h, z], -1)]
    entropies = []
    for _ in range(horizon):
        action, entropy = actor.sample(feats[-1])
        h, z, _ = wm.rssm.img_step(h, z, action)
        feats.append(torch.cat([h, z], -1))
        entropies.append(entropy)
    return torch.stack(feats), torch.stack(entropies)


def lambda_returns(rewards, values, discounts, lam=LAMBDA):
    """rewards, discounts: (H, N). values: (H+1, N). Returns (H, N).

    R_t = r_{t+1} + d_{t+1} * ((1 - lam) * v_{t+1} + lam * R_{t+1}),
    bootstrapped with R_H = v_H.
    """
    ret = values[-1]
    out = []
    for t in reversed(range(rewards.shape[0])):
        ret = rewards[t] + discounts[t] * (
            (1.0 - lam) * values[t + 1] + lam * ret)
        out.append(ret)
    return torch.stack(out[::-1])


def ac_losses(wm, actor, critic, start_h, start_z, horizon=HORIZON):
    """Compute actor and critic losses from imagined rollouts.

    start_h, start_z: (N, ...) posterior states, already detached from the
    world model graph by the caller.
    """
    feats, entropies = imagine(wm, actor, start_h, start_z, horizon)
    rewards = symexp(wm.reward_head(feats[1:]))
    conts = torch.sigmoid(wm.cont_head(feats[1:]))
    values = symexp(critic(feats))
    discounts = GAMMA * conts

    returns = lambda_returns(rewards, values, discounts)
    # Down-weight steps that follow a predicted episode end.
    weights = torch.cat([torch.ones_like(discounts[:1]),
                         torch.cumprod(discounts[:-1], 0)], 0).detach()

    actor_loss = (-(weights * returns).mean()
                  - ENTROPY_BETA * (weights * entropies).mean())

    # Critic targets use stop-gradient instead of an EMA target network: the
    # bootstrap already comes through detached returns, an EMA copy mostly
    # pays off in long large-scale runs, and one network keeps this readable.
    v_pred = critic(feats[:-1].detach())
    critic_loss = 0.5 * (weights * (v_pred
                                    - symlog(returns.detach())) ** 2).mean()

    metrics = {
        "ac/actor": actor_loss.item(),
        "ac/critic": critic_loss.item(),
        "ac/img_reward": rewards.mean().item(),
        "ac/img_return": returns.mean().item(),
        "ac/entropy": entropies.mean().item(),
    }
    return actor_loss, critic_loss, metrics
