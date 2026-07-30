"""Stage 1 training: V (VAE), M (MDN-RNN), C (controller via CMA-ES in the dream).

Usage:
    uv run python -m stage1_ha_worldmodel.train --smoke            # whole chain, tiny, CPU
    uv run python -m stage1_ha_worldmodel.train vae --device cuda
    uv run python -m stage1_ha_worldmodel.train mdnrnn --device cuda
    uv run python -m stage1_ha_worldmodel.train controller [--real]

--device applies to VAE and MDN-RNN training and to encoding. The
controller phase always runs on CPU: CMA-ES is numpy and the dream is
single-step LSTM calls, where GPU transfer overhead loses.

FULL config (the defaults below) is written for one RTX 5090 and is
documentation: only --smoke has been executed here.
  Data: 2000 episodes x 1000 steps (about 2M frames). Collection is
    CPU-bound Box2D, around 2-3 h single process; shard it across processes
    if you care.
  VAE: batch 256, 30k steps, Adam 1e-4. About 1-1.5 h. Expect recon loss
    (sum MSE per frame) to settle around 20-40 and reconstructions that keep
    road edges but blur grass texture.
  MDN-RNN: batch 128, seq len 64, 20k steps, Adam 1e-3. About 1-1.5 h.
    Expect NLL well below zero (it is a continuous density).
  Controller: CMA-ES pop 64, sigma0 0.5, 400 generations in the dream
    (minutes, it is all CPU LSTM steps). With --real, pop 64 evaluated in
    the real env is the honest but slow path: about 12 s per 1000-step
    episode per candidate, so budget hours and use fewer generations.
    Expected outcome: dream-trained controller scores modestly in the real
    env (the proxy reward caps it, see WHY.md); real-env CMA-ES is what
    reaches the 600-900 range reported at worldmodels.github.io.
Modal cost, one line: about $8-15 total (3-4 h of A100 at ~$2.50/h for
VAE+MDN-RNN plus a few CPU-hours for collection and CMA-ES).
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stage1_ha_worldmodel.cmaes import CMAES
from stage1_ha_worldmodel.dream import dream_rollout, write_dream_video
from stage1_ha_worldmodel.mdnrnn import MDNRNN, mdn_nll
from stage1_ha_worldmodel.s1_controller import Controller
from stage1_ha_worldmodel.s1_data import (frames_to_tensor, iter_episodes,
                                          load_or_collect, resize64)
from stage1_ha_worldmodel.s1_vae import ConvVAE, vae_loss

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "stage1_ha_worldmodel"

FULL = dict(episodes=2000, max_steps=1000, seed=0,
            vae_steps=30000, vae_batch=256, vae_lr=1e-4,
            mdn_steps=20000, mdn_batch=128, mdn_seq_len=64, mdn_lr=1e-3,
            cma_pop=64, cma_gens=400, cma_sigma0=0.5, dream_horizon=1000,
            dream_rollouts=4, temperature=1.15, video_horizon=300)

SMOKE = dict(episodes=2, max_steps=110, seed=7,
             vae_steps=160, vae_batch=32, vae_lr=1e-3,
             mdn_steps=200, mdn_batch=16, mdn_seq_len=16, mdn_lr=1e-3,
             cma_pop=8, cma_gens=3, cma_sigma0=0.5, dream_horizon=40,
             dream_rollouts=1, temperature=1.0, video_horizon=60)


def train_vae(cfg, data, ckpt_dir: Path, device: str = "cpu") -> ConvVAE:
    torch.manual_seed(cfg["seed"])
    vae = ConvVAE().to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg["vae_lr"])
    frames = data["frames"]
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    for step in range(cfg["vae_steps"]):
        idx = rng.integers(0, len(frames), cfg["vae_batch"])
        x = frames_to_tensor(frames[idx]).to(device)
        recon, mu, logvar = vae(x)
        loss, rec, kl = vae_loss(recon, x, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg["vae_steps"] // 5) == 0 or step == cfg["vae_steps"] - 1:
            print(f"[vae] step {step:6d} loss {loss.item():9.2f} "
                  f"recon {rec.item():9.2f} kl {kl.item():7.2f} "
                  f"({time.time() - t0:5.1f}s)")
    torch.save(vae.state_dict(), ckpt_dir / "vae.pt")
    return vae


@torch.no_grad()
def encode_data(vae: ConvVAE, data, seed: int):
    """Encode each episode to a sampled latent sequence, as in the paper:
    sampling (not just mu) gives the RNN some robustness to encoder noise."""
    torch.manual_seed(seed)
    device = next(vae.parameters()).device
    episodes = []
    for frames, actions in iter_episodes(data):
        mu, logvar = vae.encode(frames_to_tensor(frames).to(device))
        z = vae.reparameterize(mu, logvar)
        episodes.append((z.cpu().numpy().astype(np.float32), actions))
    return episodes


def train_mdnrnn(cfg, latents, ckpt_dir: Path, device: str = "cpu") -> MDNRNN:
    torch.manual_seed(cfg["seed"] + 1)
    model = MDNRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["mdn_lr"])
    L = cfg["mdn_seq_len"]
    starts = [(i, s) for i, (z, _) in enumerate(latents)
              for s in range(0, len(z) - L)]
    rng = np.random.default_rng(cfg["seed"] + 1)
    t0 = time.time()
    for step in range(cfg["mdn_steps"]):
        batch = rng.integers(0, len(starts), cfg["mdn_batch"])
        zs, acts, zn = [], [], []
        for b in batch:
            i, s = starts[b]
            z, a = latents[i]
            zs.append(z[s:s + L])
            acts.append(a[s:s + L])
            zn.append(z[s + 1:s + L + 1])
        z_in = torch.from_numpy(np.stack(zs)).to(device)
        a_in = torch.from_numpy(np.stack(acts)).to(device)
        z_next = torch.from_numpy(np.stack(zn)).to(device)
        (logpi, mu, logstd), _ = model(z_in, a_in)
        loss = mdn_nll(logpi, mu, logstd, z_next)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, cfg["mdn_steps"] // 5) == 0 or step == cfg["mdn_steps"] - 1:
            print(f"[mdnrnn] step {step:6d} nll {loss.item():8.4f} "
                  f"({time.time() - t0:5.1f}s)")
    torch.save(model.state_dict(), ckpt_dir / "mdnrnn.pt")
    return model


@torch.no_grad()
def real_return(controller, vae, mdnrnn, seed: int, max_steps: int) -> float:
    env = gym.make("CarRacing-v3")
    obs, _ = env.reset(seed=seed)
    hidden = None
    h = torch.zeros(1, mdnrnn.hidden_dim)
    total = 0.0
    for _ in range(max_steps):
        x = frames_to_tensor(resize64(obs)[None])
        mu, _ = vae.encode(x)
        a = controller(mu, h)
        obs, r, terminated, truncated, _ = env.step(
            a.view(-1).numpy().astype(np.float32))
        total += float(r)
        _, hidden = mdnrnn(mu.view(1, 1, -1), a.view(1, 1, -1), hidden)
        h = hidden[0].view(1, -1)
        if terminated or truncated:
            break
    env.close()
    return total


@torch.no_grad()
def encode_first_frames(vae, data) -> np.ndarray:
    """Dream start states: the encoded first frame of each episode."""
    device = next(vae.parameters()).device
    first = np.stack([f[0] for f, _ in iter_episodes(data)])
    mu, _ = vae.encode(frames_to_tensor(first).to(device))
    return mu.cpu().numpy().astype(np.float32)


def train_controller(cfg, vae, mdnrnn, z_starts, ckpt_dir: Path,
                     temperature: float, real: bool = False):
    # CMA-ES is numpy and the dream is single-step LSTM calls: CPU work,
    # regardless of where the earlier phases trained. See module docstring.
    vae.cpu().eval()
    mdnrnn.cpu().eval()
    T = temperature
    controller = Controller(z_dim=mdnrnn.z_dim, hidden_dim=mdnrnn.hidden_dim)
    print(f"controller parameters: {controller.param_count()}")
    es = CMAES(np.zeros(controller.param_count()), cfg["cma_sigma0"],
               popsize=cfg["cma_pop"], seed=cfg["seed"] + 2)
    rng = np.random.default_rng(cfg["seed"] + 2)
    best_fit, best_params = np.inf, None
    for gen in range(cfg["cma_gens"]):
        candidates = es.ask()
        fits = []
        for c, params in enumerate(candidates):
            controller.set_flat_params(params)
            if real:
                ret = real_return(controller, vae, mdnrnn,
                                  seed=cfg["seed"] + 100 + gen,
                                  max_steps=cfg["max_steps"])
            else:
                rets = []
                for k in range(cfg["dream_rollouts"]):
                    z0 = torch.from_numpy(
                        z_starts[rng.integers(0, len(z_starts))])
                    gen_t = torch.Generator().manual_seed(
                        cfg["seed"] + 10000 + gen * 1000 + c * 10 + k)
                    out = dream_rollout(mdnrnn, controller, z0,
                                        cfg["dream_horizon"], temperature=T,
                                        generator=gen_t)
                    rets.append(out["ret"])
                ret = float(np.mean(rets))
            fits.append(-ret)
        es.tell(candidates, fits)
        i = int(np.argmin(fits))
        if fits[i] < best_fit:
            best_fit, best_params = fits[i], candidates[i].copy()
        where = "real" if real else f"dream(T={T})"
        print(f"[controller] gen {gen:3d} best return {-fits[i]:8.3f} "
              f"mean {-float(np.mean(fits)):8.3f} ({where})")
    controller.set_flat_params(best_params)
    np.savez(ckpt_dir / "controller.npz", params=best_params,
             fitness=best_fit, temperature=T, real=real)
    return controller


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["vae", "mdnrnn", "controller", "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny everything, CPU, under two minutes")
    ap.add_argument("--real", action="store_true",
                    help="evaluate CMA-ES candidates in the real env "
                         "instead of the dream")
    ap.add_argument("--temperature", type=float, default=None,
                    help="dream temperature for controller training and video")
    ap.add_argument("--device", default="cpu",
                    help="device for VAE/MDN-RNN training, e.g. cuda")
    args = ap.parse_args()

    # Small CPU models scale badly past a few threads.
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    cfg = SMOKE if args.smoke else FULL
    tag = "smoke" if args.smoke else "full"
    ckpt_dir = DATA_DIR / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    data = load_or_collect(DATA_DIR, cfg["episodes"], cfg["max_steps"],
                           cfg["seed"])
    print(f"dataset: {len(data['frames'])} frames, "
          f"{len(data['ep_lens'])} episodes")

    def load_vae():
        vae = ConvVAE()
        vae.load_state_dict(torch.load(ckpt_dir / "vae.pt", map_location="cpu",
                                       weights_only=True))
        return vae

    def load_mdn():
        m = MDNRNN()
        m.load_state_dict(torch.load(ckpt_dir / "mdnrnn.pt", map_location="cpu",
                                     weights_only=True))
        return m

    if args.stage in ("vae", "all"):
        vae = train_vae(cfg, data, ckpt_dir, device=args.device)
    else:
        vae = load_vae()

    if args.stage in ("mdnrnn", "all"):
        vae = vae.to(args.device)
        latents = encode_data(vae, data, cfg["seed"])
        mdnrnn = train_mdnrnn(cfg, latents, ckpt_dir, device=args.device)
    elif args.stage == "controller":
        mdnrnn = load_mdn()
    else:
        mdnrnn = None

    if args.stage in ("controller", "all"):
        T = cfg["temperature"] if args.temperature is None else args.temperature
        z_starts = encode_first_frames(vae, data)
        train_controller(cfg, vae, mdnrnn, z_starts, ckpt_dir,
                         temperature=T, real=args.real)
        frames, actions = next(iter_episodes(data))
        n = min(cfg["video_horizon"], len(frames))
        path = write_dream_video(vae, mdnrnn, frames[:n], actions[:n],
                                 ckpt_dir / f"dream_vs_real_T{T}.gif",
                                 temperature=T, seed=cfg["seed"])
        print(f"dream video: {path}")

    print("done")


if __name__ == "__main__":
    main()
