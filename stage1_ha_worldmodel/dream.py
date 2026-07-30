"""Closed-loop rollouts entirely in latent space, plus dream videos.

A dream step: the controller reads (z_t, h_t), picks a_t, the MDN-RNN
advances its hidden state on (z_t, a_t), and z_{t+1} is sampled from the
mixture at the chosen temperature. No pixels are touched until you decode
frames for a video.

Proxy reward, stated honestly: the world model predicts no reward, so the
dream cannot score "tiles visited" like the real CarRacing does. We use

    r_t = gas_t - brake_t - 0.1 * |steer_t| - 0.05 * mean(z_{t+1}^2)

which only encodes "keep the throttle on, do not thrash the wheel, and keep
the dream near the training distribution" (under the VAE prior, mean(z^2)
is about 1 on-distribution and grows when the dream drifts off-manifold).
It is a stand-in, not the real objective; the dream-vs-real gap this opens
is discussed in WHY.md and measured in the exercises.
"""

import numpy as np
import torch
import imageio

from stage1_ha_worldmodel.mdnrnn import mdn_sample
from stage1_ha_worldmodel.s1_data import frames_to_tensor


def proxy_reward(action: torch.Tensor, z_next: torch.Tensor) -> float:
    steer, gas, brake = (float(action[0]), float(action[1]), float(action[2]))
    off_manifold = float((z_next ** 2).mean())
    return gas - brake - 0.1 * abs(steer) - 0.05 * off_manifold


def mdn_step(mdnrnn, z, a, hidden, temperature, generator):
    """One latent advance: run the RNN on (z_t, a_t), sample z_{t+1}.

    z: (1, 1, z_dim), a: (1, 1, action_dim). Returns (z_next (1, 1, z_dim), hidden).
    """
    (logpi, mu, logstd), hidden = mdnrnn(z, a, hidden)
    z_next = mdn_sample(logpi[:, -1], mu[:, -1], logstd[:, -1],
                        temperature=temperature, generator=generator)
    return z_next.view(1, 1, -1), hidden


@torch.no_grad()
def dream_rollout(mdnrnn, controller, z0: torch.Tensor, horizon: int,
                  temperature: float = 1.0, generator=None) -> dict:
    """Roll the dream forward from latent z0. Returns zs (horizon+1, z_dim),
    actions (horizon, 3), rewards (horizon,), and the summed proxy return."""
    z = z0.view(1, 1, -1).float()
    h = torch.zeros(1, mdnrnn.hidden_dim)
    hidden = None
    zs, actions, rewards = [z.view(-1).clone()], [], []
    for _ in range(horizon):
        a = controller(z.view(1, -1), h)
        z_next, hidden = mdn_step(mdnrnn, z, a.view(1, 1, -1), hidden,
                                  temperature, generator)
        rewards.append(proxy_reward(a.view(-1), z_next))
        actions.append(a.view(-1).clone())
        z = z_next
        h = hidden[0].view(1, -1)
        zs.append(z.view(-1).clone())
    return dict(zs=torch.stack(zs), actions=torch.stack(actions),
                rewards=np.asarray(rewards), ret=float(np.sum(rewards)))


@torch.no_grad()
def write_dream_video(vae, mdnrnn, frames_u8: np.ndarray, actions: np.ndarray,
                      out_path, temperature: float = 1.0, seed: int = 0,
                      fps: int = 20) -> str:
    """Side-by-side GIF: real frames | VAE reconstruction | dream.

    The dream starts from the encoding of the first real frame and is driven
    by the same recorded actions, so any divergence you see in the right
    panel is the dynamics model drifting, not a different policy. The middle
    panel isolates VAE blur from that drift.
    """
    device = next(vae.parameters()).device
    mu, _ = vae.encode(frames_to_tensor(frames_u8).to(device))
    recon = vae.decode(mu)

    gen = torch.Generator(device=device.type).manual_seed(seed)
    z = mu[0].view(1, 1, -1)
    hidden = None
    dream_zs = [z.view(-1).clone()]
    for t in range(len(actions) - 1):
        a = torch.from_numpy(actions[t]).float().view(1, 1, -1).to(device)
        z, hidden = mdn_step(mdnrnn, z, a, hidden, temperature, gen)
        dream_zs.append(z.view(-1).clone())
    dream_frames = vae.decode(torch.stack(dream_zs))

    def to_u8(t):
        return (t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    recon_u8, dream_u8 = to_u8(recon), to_u8(dream_frames)
    n = len(dream_u8)
    panels = np.concatenate([frames_u8[:n], recon_u8[:n], dream_u8], axis=2)
    panels = panels.repeat(2, axis=1).repeat(2, axis=2)  # 2x upscale
    imageio.mimsave(str(out_path), list(panels), duration=1000.0 / fps, loop=0)
    return str(out_path)
