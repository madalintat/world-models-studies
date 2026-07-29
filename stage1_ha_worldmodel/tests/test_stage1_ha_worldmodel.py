"""Fast checks for stage 1. No environment, no prior training runs needed."""

import numpy as np
import torch

# The models here are tiny; extra threads only add sync overhead.
torch.set_num_threads(2)

from stage1_ha_worldmodel.cmaes import CMAES
from stage1_ha_worldmodel.dream import dream_rollout
from stage1_ha_worldmodel.mdnrnn import MDNRNN, mdn_nll, mdn_sample
from stage1_ha_worldmodel.s1_controller import Controller
from stage1_ha_worldmodel.s1_vae import ConvVAE, vae_loss


def test_vae_shapes_and_loss():
    torch.manual_seed(0)
    vae = ConvVAE()
    x = torch.rand(4, 3, 64, 64)
    recon, mu, logvar = vae(x)
    assert recon.shape == (4, 3, 64, 64)
    assert mu.shape == (4, 32) and logvar.shape == (4, 32)
    loss, rec, kl = vae_loss(recon, x, mu, logvar)
    assert torch.isfinite(loss)


def test_controller_param_count_matches_formula():
    z_dim, hidden_dim = 32, 256
    c = Controller(z_dim=z_dim, hidden_dim=hidden_dim)
    assert c.param_count() == (z_dim + hidden_dim + 1) * 3 == 867
    out = c(torch.randn(5, z_dim), torch.randn(5, hidden_dim))
    assert out.shape == (5, 3)
    assert (out[:, 0].abs() <= 1).all()
    assert (out[:, 1:] >= 0).all() and (out[:, 1:] <= 1).all()


def test_controller_flat_params_roundtrip():
    c = Controller()
    flat = np.random.default_rng(0).standard_normal(c.param_count())
    c.set_flat_params(flat)
    assert np.allclose(c.get_flat_params(), flat, atol=1e-6)


def _synthetic_sequences(rng, batch, length, z_dim, a_dim):
    """Learnable dynamics: z rotates a bit each step and the action shifts it."""
    theta = 0.2
    rot = np.eye(z_dim)
    rot[0, 0] = rot[1, 1] = np.cos(theta)
    rot[0, 1], rot[1, 0] = -np.sin(theta), np.sin(theta)
    z = rng.standard_normal((batch, z_dim))
    a = rng.standard_normal((batch, length, a_dim)).astype(np.float32)
    zs = []
    for t in range(length + 1):
        zs.append(z.copy())
        if t < length:
            z = z @ rot.T
            z[:, :a_dim] += 0.3 * a[:, t]
            z = z + 0.02 * rng.standard_normal(z.shape)
    zs = np.stack(zs, axis=1).astype(np.float32)
    return zs[:, :-1], a, zs[:, 1:]


def test_mdn_nll_decreases_on_learnable_sequence():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = MDNRNN(z_dim=4, action_dim=2, hidden_dim=32, n_gaussians=5)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    z, a, z_next = _synthetic_sequences(rng, batch=32, length=20, z_dim=4, a_dim=2)
    z, a, z_next = map(torch.from_numpy, (z, a, z_next))
    losses = []
    for _ in range(250):
        (logpi, mu, logstd), _ = model(z, a)
        loss = mdn_nll(logpi, mu, logstd, z_next)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.3, (losses[0], losses[-1])


def test_temperature_increases_sampling_variance_monotonically():
    torch.manual_seed(0)
    n = 20000
    logpi = torch.tensor([0.0, 1.0, -1.0, 0.5, 0.0]).expand(n, 1, 5)
    mu = torch.tensor([-2.0, 0.0, 2.0, 1.0, -1.0]).expand(n, 1, 5)
    logstd = torch.full((n, 1, 5), -1.0)
    variances = []
    for t_i, temp in enumerate([0.3, 1.0, 2.0]):
        gen = torch.Generator().manual_seed(42 + t_i)
        s = mdn_sample(logpi, mu, logstd, temperature=temp, generator=gen)
        variances.append(s.var().item())
    assert variances[0] < variances[1] < variances[2], variances


def test_cmaes_optimizes_quadratic_to_near_zero():
    target = np.arange(8, dtype=np.float64) / 8.0

    def f(x):
        return float(np.sum((x - target) ** 2))

    es = CMAES(np.zeros(8), sigma0=0.5, seed=3)
    best = np.inf
    for _ in range(200):
        xs = es.ask()
        fits = [f(x) for x in xs]
        es.tell(xs, fits)
        best = min(best, min(fits))
    assert best < 1e-8, best


def test_dream_rollout_shapes_and_determinism():
    torch.manual_seed(0)
    mdn = MDNRNN(z_dim=8, action_dim=3, hidden_dim=16, n_gaussians=5)
    ctrl = Controller(z_dim=8, hidden_dim=16)
    z0 = torch.randn(8)
    out1 = dream_rollout(mdn, ctrl, z0, horizon=12, temperature=1.0,
                         generator=torch.Generator().manual_seed(1))
    out2 = dream_rollout(mdn, ctrl, z0, horizon=12, temperature=1.0,
                         generator=torch.Generator().manual_seed(1))
    assert out1["zs"].shape == (13, 8)
    assert out1["actions"].shape == (12, 3)
    assert out1["rewards"].shape == (12,)
    assert torch.allclose(out1["zs"], out2["zs"])
