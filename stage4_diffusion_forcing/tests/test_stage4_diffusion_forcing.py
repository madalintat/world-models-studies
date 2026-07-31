import numpy as np
import pytest
import torch

from stage4_diffusion_forcing.flow import (
    euler_beta,
    euler_mix,
    flow_matching_loss,
    interpolate,
    loss_weight,
    make_ladder,
    sample_frame_taus,
    scheduling_matrix,
)
from stage4_diffusion_forcing.s4_latent_ae import LatentAE
from stage4_diffusion_forcing.s4_model import DynamicsTransformer
from stage4_diffusion_forcing.sampling4 import drift_curve, rollout, rollout_block

torch.manual_seed(0)


def tiny_model(max_t=8, injection="token"):
    torch.manual_seed(1)
    return DynamicsTransformer(
        latent_dim=4, n_latent_tokens=16, d_model=32, n_heads=2, depth=4,
        max_t=max_t, injection=injection,
    )


def pyramid_or_skip(k_steps, n_frames, stagger=1):
    try:
        return scheduling_matrix("pyramid", k_steps, n_frames, stagger)
    except NotImplementedError:
        pytest.skip("_pyramid_schedule is yours to write: see exercises.md")


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


def test_logit_normal_tau_concentrates_on_mid_noise():
    g = torch.Generator().manual_seed(0)
    uni = sample_frame_taus(4096, 8, generator=g)
    ln = sample_frame_taus(4096, 8, scheme="logit_normal", generator=g)
    assert ln.shape == (4096, 8)
    assert ln.min() > 0.0 and ln.max() < 1.0
    # sigmoid of a symmetric normal is symmetric about 0.5 ...
    assert abs(ln.mean().item() - 0.5) < 0.02
    # ... but tighter than uniform, which is the whole point: fewer samples
    # wasted on the trivially easy ends of the noise range.
    assert ln.std().item() < uni.std().item()
    with pytest.raises(ValueError):
        sample_frame_taus(2, 2, scheme="nope")


def test_scheduling_matrix_corners():
    k, n = 3, 4

    full = scheduling_matrix("full_sequence", k, n)
    assert full.shape == (k + 1, n)
    # every frame moves together, so all columns are identical
    assert (full == full[:, :1]).all()

    seq = scheduling_matrix("sequential", k, n)
    assert seq.shape == (k * n + 1, n)
    # frame f is still pure noise until frame f-1 is finished
    for f in range(1, n):
        assert seq[k * f, f] == 0
        assert seq[k * f, f - 1] == k

    for m in (full, seq):
        assert (m[0] == 0).all(), "row 0 is pure noise"
        assert (m[-1] == k).all(), "last row is clean"
        assert (m.diff(dim=0) >= 0).all(), "noise level never goes backwards"

    with pytest.raises(ValueError):
        scheduling_matrix("nope", k, n)


def test_pyramid_schedule_contract():
    k, n = 2, 3
    m = pyramid_or_skip(k, n)
    assert m.shape == (k + n, n)
    assert (m[0] == 0).all()
    assert (m[-1] == k).all()
    assert (m.diff(dim=0) >= 0).all()
    expected = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, 2, 1], [2, 2, 2]])
    assert torch.equal(m, expected)


def test_pyramid_stagger_interpolates_between_the_corners():
    """stagger 0 collapses to full sequence, stagger k to sequential."""
    k, n = 3, 4
    assert torch.equal(pyramid_or_skip(k, n, 0)[: k + 1],
                       scheduling_matrix("full_sequence", k, n))
    assert torch.equal(pyramid_or_skip(k, n, k), scheduling_matrix("sequential", k, n))


def test_euler_beta_matches_the_ladder():
    taus, betas = make_ladder(4)
    assert torch.allclose(euler_beta(taus[:-1], taus[1:]), betas)
    # a frame that does not move on this row must be left untouched
    tau = torch.tensor([0.5])
    assert torch.allclose(euler_beta(tau, tau), torch.ones(1))
    z, x_hat = torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2)
    assert torch.allclose(euler_mix(z, x_hat, euler_beta(tau, tau)), z)


def test_block_rollout_shapes_and_call_counts():
    model = tiny_model(max_t=8)
    prefill = torch.randn(1, 4, 16, 4)
    actions = torch.randn(1, 8, 3)
    for mode in ("full_sequence", "sequential"):
        out = rollout_block(
            model, prefill, actions, horizon=4, k_steps=2, mode=mode,
            generator=torch.Generator().manual_seed(0),
        )
        assert out.shape == (1, 4, 16, 4)
        assert torch.isfinite(out).all()
    # a block longer than the model's window is a bug, not a silent truncation
    with pytest.raises(AssertionError):
        rollout_block(model, prefill, actions, horizon=8, k_steps=2)


def test_pyramid_is_cheaper_than_sequential_for_the_same_block():
    """The reason to care: rows are model calls."""
    k, n = 4, 8
    pyramid_rows = pyramid_or_skip(k, n).shape[0]
    assert pyramid_rows < scheduling_matrix("sequential", k, n).shape[0]
    model = tiny_model(max_t=12)
    out = rollout_block(
        model, torch.randn(1, 4, 16, 4), torch.randn(1, 12, 3), horizon=8,
        k_steps=k, mode="pyramid",
        generator=torch.Generator().manual_seed(0),
    )
    assert out.shape == (1, 8, 16, 4)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("injection", ["token", "additive", "film"])
def test_action_injection_variants_learn_from_the_action(injection):
    """A world model that cannot learn from its action is not a world model.

    This checks the gradient rather than the output, in the same spirit as
    stage 0's test that gradient reaches fc_logvar through the reparam
    trick. FiLM needs two steps to get there and that is worth knowing:
    its modulation head is zero-initialized (the DiT convention, so the
    block starts as an identity map and cannot destabilize early training),
    and since FiLM is the action's *only* path into the network, a zero
    modulation weight means the action embedding receives exactly zero
    gradient on step 0. The head itself does get gradient, so the path
    opens on step 1. With "token" or "additive" the action reaches the
    latents directly and there is no such delay.
    """
    model = tiny_model(injection=injection)
    z = torch.randn(2, 6, 16, 4)
    a = torch.randn(2, 6, 3)
    tau = torch.rand(2, 6)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    out = model(z, a, tau)
    assert out.shape == (2, 6, 16, 4)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()

    if injection == "film":
        heads = [b.film.weight.grad for b in model.blocks]
        assert all(g is not None and g.abs().sum() > 0 for g in heads)
        assert model.action_in.weight.grad.abs().sum() == 0, "see the docstring"
        opt.step()
        opt.zero_grad()
        model(z, a, tau).pow(2).mean().backward()

    assert model.action_in.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("injection", ["token", "additive"])
def test_non_zero_init_injections_change_the_output_immediately(injection):
    model = tiny_model(injection=injection).eval()
    z, tau = torch.randn(2, 6, 16, 4), torch.rand(2, 6)
    with torch.no_grad():
        out = model(z, torch.randn(2, 6, 3), tau)
        other = model(z, torch.randn(2, 6, 3) + 5.0, tau)
    assert not torch.allclose(out, other, atol=1e-4)


def test_token_injection_costs_a_sequence_slot():
    """The trade-off the ablation is about, made concrete."""
    tok, add = tiny_model(injection="token"), tiny_model(injection="additive")
    assert tok.n_prefix == 2 and add.n_prefix == 1
    assert tok.space_pos.shape[2] == add.space_pos.shape[2] + 1
    # additive reuses the action embedding it already had, so it is free
    assert sum(p.numel() for p in add.parameters()) < sum(
        p.numel() for p in tok.parameters()
    )
    # film pays for a modulation head in every block
    film = tiny_model(injection="film")
    assert sum(p.numel() for p in film.parameters()) > sum(
        p.numel() for p in add.parameters()
    )


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
