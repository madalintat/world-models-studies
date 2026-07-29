import numpy as np
import torch

from stage4_diffusion_forcing.flow import (
    euler_mix,
    flow_matching_loss,
    interpolate,
    loss_weight,
    make_ladder,
    sample_frame_taus,
)
from stage4_diffusion_forcing.s4_latent_ae import LatentAE
from stage4_diffusion_forcing.s4_model import DynamicsTransformer
from stage4_diffusion_forcing.sampling4 import drift_curve, rollout

torch.manual_seed(0)


def tiny_model(max_t=8):
    torch.manual_seed(1)
    return DynamicsTransformer(
        latent_dim=4, n_latent_tokens=16, d_model=32, n_heads=2, depth=4, max_t=max_t
    )


def test_interpolation_endpoints():
    noise = torch.randn(2, 3, 16, 4)
    x = torch.randn(2, 3, 16, 4)
    z0 = interpolate(noise, x, torch.zeros(2, 3))
    z1 = interpolate(noise, x, torch.ones(2, 3))
    assert torch.allclose(z0, noise)
    assert torch.allclose(z1, x)


def test_per_frame_tau_independence():
    g = torch.Generator().manual_seed(0)
    tau = sample_frame_taus(512, 16, generator=g)
    assert tau.shape == (512, 16)
    assert tau.min() >= 0.0 and tau.max() <= 1.0
    # uniform on [0, 1]: mean 0.5, std about 0.289
    assert abs(tau.mean().item() - 0.5) < 0.02
    assert abs(tau.std().item() - 0.2887) < 0.02
    # frames within a sequence are not tied together
    corr = np.corrcoef(tau[:, 0].numpy(), tau[:, 1].numpy())[0, 1]
    assert abs(corr) < 0.1


def test_loss_weight_schemes():
    tau = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(loss_weight(tau, "none"), torch.ones(3))
    assert torch.allclose(loss_weight(tau, "ramp"), torch.tensor([0.1, 0.55, 1.0]))
    v = loss_weight(tau, "v_space")
    assert torch.allclose(v[:2], torch.tensor([1.0, 4.0]))
    assert v[2] > 1e5  # clamped, large but finite near tau = 1


def test_training_step_decreases_loss_on_repeated_batch():
    model = tiny_model()
    g = torch.Generator().manual_seed(2)
    z = torch.randn(4, 6, 16, 4, generator=g)
    a = torch.randn(4, 6, 3, generator=g)
    tau = sample_frame_taus(4, 6, generator=g)
    noise = torch.randn(z.shape, generator=g)
    z_tau = interpolate(noise, z, tau)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(25):
        loss = flow_matching_loss(model(z_tau, a, tau), z, tau, weighting="ramp")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8


def test_k1_ladder_equals_one_shot_x_prediction():
    taus, betas = make_ladder(1)
    assert torch.allclose(taus, torch.tensor([0.0, 1.0]))
    assert torch.allclose(betas, torch.tensor([0.0]))
    # so one Euler step returns the x-prediction exactly, whatever z was
    z = torch.randn(2, 1, 16, 4)
    x_hat = torch.randn(2, 1, 16, 4)
    assert torch.allclose(euler_mix(z, x_hat, betas[0]), x_hat)

    # end to end: a K=1 rollout of one frame equals a single model call on
    # pure noise at tau = 0
    model = tiny_model()
    prefill = torch.randn(2, 3, 16, 4)
    actions = torch.randn(2, 4, 3)
    out = rollout(
        model, prefill, actions, horizon=1, k_steps=1,
        generator=torch.Generator().manual_seed(7),
    )
    g = torch.Generator().manual_seed(7)
    z0 = torch.randn((2, 1, 16, 4), generator=g)
    tau_vec = torch.tensor([1.0, 1.0, 1.0, 0.0]).expand(2, -1)
    with torch.no_grad():
        manual = model(torch.cat([prefill, z0], dim=1), actions, tau_vec)[:, -1:]
    assert torch.allclose(out, manual, atol=1e-6)


def test_causal_time_attention_does_not_leak_future():
    model = tiny_model().eval()
    z = torch.randn(2, 8, 16, 4)
    a = torch.randn(2, 8, 3)
    tau = torch.rand(2, 8)
    with torch.no_grad():
        base = model(z, a, tau)
        z2 = z.clone()
        z2[:, 5:] += torch.randn_like(z2[:, 5:]) * 10.0
        pert = model(z2, a, tau)
    assert torch.allclose(base[:, :5], pert[:, :5], atol=1e-5)
    assert not torch.allclose(base[:, 5:], pert[:, 5:], atol=1e-2)


def test_ae_shapes_roundtrip():
    ae = LatentAE(latent_dim=8, base=8)
    x = torch.rand(2, 3, 64, 64)
    z = ae.encode(x)
    assert z.shape == (2, 8, 8, 8)
    tokens = LatentAE.to_tokens(z)
    assert tokens.shape == (2, 64, 8)
    assert torch.allclose(LatentAE.from_tokens(tokens), z)
    recon = ae.decode(z)
    assert recon.shape == (2, 3, 64, 64)
    assert recon.min() >= 0.0 and recon.max() <= 1.0


def test_drift_curve_finite():
    pred = torch.rand(6, 3, 64, 64)
    gt = torch.rand(6, 3, 64, 64)
    curve = drift_curve(pred, gt)
    assert curve.shape == (6,)
    assert np.isfinite(curve).all()
    same = drift_curve(gt, gt)
    assert np.isfinite(same).all()  # identical frames must not blow up


def test_rollout_shapes_and_window():
    model = tiny_model(max_t=8)
    prefill = torch.randn(1, 4, 16, 4)
    actions = torch.randn(1, 14, 3)
    out = rollout(
        model, prefill, actions, horizon=10, k_steps=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert out.shape == (1, 10, 16, 4)
    assert torch.isfinite(out).all()
