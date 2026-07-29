import numpy as np
import pytest
import torch

# Tiny tensors run far faster on one thread than on the default pool.
torch.set_num_threads(1)

from stage2_dreamer.buffer import EpisodeBuffer
from stage2_dreamer.rssm import RSSM, kl_loss
from stage2_dreamer.s2_ac import Actor, Critic, ac_losses, imagine
from stage2_dreamer.s2_wm import WorldModel, symexp, symlog


def tiny_wm():
    torch.manual_seed(0)
    return WorldModel(action_dim=3, depth=4, embed_dim=32, deter_dim=32,
                      hidden_dim=32, n_vars=4, n_classes=4)


def random_batch(B=3, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "obs": torch.randint(0, 256, (B, T, 64, 64, 3), generator=g,
                             dtype=torch.uint8),
        "action": torch.rand(B, T, 3, generator=g) * 2 - 1,
        "reward": torch.randn(B, T, generator=g),
        "cont": torch.ones(B, T),
    }


def test_rssm_step_shapes():
    torch.manual_seed(0)
    rssm = RSSM(action_dim=3, embed_dim=16, deter_dim=24, hidden_dim=24,
                n_vars=4, n_classes=5)
    B = 6
    h, z = rssm.initial(B, "cpu")
    assert h.shape == (B, 24) and z.shape == (B, 20)
    a, e = torch.randn(B, 3), torch.randn(B, 16)
    h2, z2, pl, ql = rssm.obs_step(h, z, a, e)
    assert h2.shape == (B, 24) and z2.shape == (B, 20)
    assert pl.shape == (B, 4, 5) and ql.shape == (B, 4, 5)
    # z is a straight-through one-hot sample: one 1 per variable.
    assert torch.allclose(z2.view(B, 4, 5).sum(-1), torch.ones(B, 4))
    h3, z3, pl3 = rssm.img_step(h2, z2, a)
    assert h3.shape == (B, 24) and z3.shape == (B, 20)
    T = 7
    hs, zs, pls, qls = rssm.observe(torch.randn(B, T, 16),
                                    torch.randn(B, T, 3))
    assert hs.shape == (B, T, 24) and zs.shape == (B, T, 20)
    assert pls.shape == (B, T, 4, 5) and qls.shape == (B, T, 4, 5)


def test_kl_balancing_direction():
    # With alpha 0.8 the prior side must receive clearly more gradient than
    # the posterior side.
    torch.manual_seed(1)
    prior = (torch.randn(16, 8, 8) * 2).requires_grad_()
    post = (torch.randn(16, 8, 8) * 2).requires_grad_()
    loss, kl_value = kl_loss(post, prior)
    assert kl_value.item() > 1.0, "test needs KL above the free bits floor"
    loss.backward()
    assert prior.grad.norm() > 2 * post.grad.norm()


def test_free_bits_kill_gradient():
    torch.manual_seed(2)
    logits = torch.randn(4, 4, 4)
    prior = logits.clone().requires_grad_()
    post = logits.clone().requires_grad_()
    loss, kl_value = kl_loss(post, prior)
    assert kl_value.item() < 1.0
    loss.backward()
    assert prior.grad.abs().max() == 0
    assert post.grad.abs().max() == 0


def test_symlog_roundtrip():
    x = torch.linspace(-50, 50, 101)
    assert torch.allclose(symlog(symexp(x)), x, atol=1e-4)
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_imagination_rollout_finite():
    wm = tiny_wm()
    actor = Actor(wm.feat_dim, 3, 32)
    h, z = wm.rssm.initial(5, "cpu")
    H = 15
    feats, entropies = imagine(wm, actor, h, z, H)
    assert feats.shape == (H + 1, 5, wm.feat_dim)
    assert entropies.shape == (H, 5)
    assert torch.isfinite(feats).all() and torch.isfinite(entropies).all()


def test_buffer_sample_shapes():
    buf = EpisodeBuffer(capacity_steps=1000, seq_len=16, seed=0)
    T = 40
    buf.add_episode(np.zeros((T, 64, 64, 3), np.uint8),
                    np.zeros((T, 3), np.float32),
                    np.zeros(T, np.float32), np.ones(T, np.float32))
    batch = buf.sample(5)
    assert batch["obs"].shape == (5, 16, 64, 64, 3)
    assert batch["obs"].dtype == torch.uint8
    assert batch["action"].shape == (5, 16, 3)
    assert batch["reward"].shape == (5, 16)
    assert batch["cont"].shape == (5, 16)


def test_combined_train_step_reduces_wm_loss():
    wm = tiny_wm()
    actor = Actor(wm.feat_dim, 3, 32)
    critic = Critic(wm.feat_dim, 32)
    batch = random_batch()
    wm_opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)

    first_loss, last_loss = None, None
    for step in range(20):
        wm_opt.zero_grad()
        total, metrics, (hs, zs) = wm.loss(batch)
        total.backward()
        wm_opt.step()
        if first_loss is None:
            first_loss = metrics["wm/total"]
        last_loss = metrics["wm/total"]

        actor_loss, critic_loss, ac_metrics = ac_losses(
            wm, actor, critic, hs.flatten(0, 1), zs.flatten(0, 1), horizon=5)
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        for v in {**metrics, **ac_metrics}.values():
            assert np.isfinite(v)

    assert last_loss < first_loss
