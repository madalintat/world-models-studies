import numpy as np
import torch

from stage0_compression.models import ConvAE, ConvVAE, vae_loss
from stage0_compression.viz import interpolation_strip, latent_traversal, recon_grid


def _random_frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 64, 64, 3), dtype=np.uint8)


def test_ae_shapes():
    model = ConvAE(latent_dim=32)
    x = torch.rand(4, 3, 64, 64)
    x_hat, z = model(x)
    assert x_hat.shape == (4, 3, 64, 64)
    assert z.shape == (4, 32)
    assert x_hat.min() >= 0 and x_hat.max() <= 1


def test_vae_shapes():
    model = ConvVAE(latent_dim=32)
    x = torch.rand(4, 3, 64, 64)
    mu, logvar = model.encode(x)
    assert mu.shape == (4, 32)
    assert logvar.shape == (4, 32)
    x_hat, mu, logvar, z = model(x)
    assert x_hat.shape == (4, 3, 64, 64)
    assert z.shape == (4, 32)


def test_vae_loss_decreases():
    torch.manual_seed(0)
    model = ConvVAE(latent_dim=8)
    x = torch.rand(16, 3, 64, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        x_hat, mu, logvar, _ = model(x)
        loss, _, _ = vae_loss(x, x_hat, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_reparam_gradient_flows():
    model = ConvVAE(latent_dim=8)
    x = torch.rand(2, 3, 64, 64)
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    z.pow(2).sum().backward()
    # If the sample were drawn non-differentiably, these grads would be None.
    assert model.fc_mu.weight.grad is not None
    assert model.fc_logvar.weight.grad is not None
    assert model.fc_logvar.weight.grad.abs().sum() > 0


def test_traversal_output_shape():
    model = ConvVAE(latent_dim=32)
    frame = _random_frames(1)[0]
    img = latent_traversal(model, frame, dims=[0, 3, 7], span=2.0, steps=5)
    assert img.shape == (3 * 64, 5 * 64, 3)
    assert img.dtype == np.uint8


def test_interpolation_and_grid_shapes():
    model = ConvAE(latent_dim=32)
    frames = _random_frames(6)
    strip = interpolation_strip(model, frames[0], frames[1], steps=6)
    assert strip.shape == (64, 6 * 64, 3)
    grid = recon_grid(model, frames)
    assert grid.shape == (2 * 64, 6 * 64, 3)
    assert grid.dtype == np.uint8
