"""ConvAE and ConvVAE for 64x64x3 CarRacing frames.

Architecture follows Ha and Schmidhuber (2018): four stride-2 convs down,
four transposed convs up, latent dimension 32 by default. Input and output
are float tensors in [0, 1] with shape (B, 3, 64, 64).
"""

import torch
import torch.nn as nn


class _Encoder(nn.Module):
    """Conv trunk: (B, 3, 64, 64) -> (B, 1024) flat features."""

    OUT_FEATURES = 256 * 2 * 2

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class _Decoder(nn.Module):
    """(B, latent_dim) -> (B, 3, 64, 64) in [0, 1]."""

    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        # Kernel sizes 5,5,6,6 land exactly on 64x64 from a 1x1 spatial start.
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 1024, 1, 1)
        return self.net(h)


class ConvAE(nn.Module):
    """Plain autoencoder: deterministic bottleneck, no distribution."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _Encoder()
        self.fc_z = nn.Linear(_Encoder.OUT_FEATURES, latent_dim)
        self.decoder = _Decoder(latent_dim)

    def encode(self, x):
        return self.fc_z(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class ConvVAE(nn.Module):
    """VAE: encode returns (mu, logvar), forward decodes a reparameterized sample."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = _Encoder()
        self.fc_mu = nn.Linear(_Encoder.OUT_FEATURES, latent_dim)
        self.fc_logvar = nn.Linear(_Encoder.OUT_FEATURES, latent_dim)
        self.decoder = _Decoder(latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps keeps the sample differentiable w.r.t. mu and logvar.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    """Returns (total, recon, kl). Recon is summed squared error per image,
    KL is the closed form against a unit Gaussian, both averaged over the batch.
    Summing (not averaging) over pixels keeps the two terms on comparable scales."""
    recon = ((x_hat - x) ** 2).sum(dim=(1, 2, 3)).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return recon + beta * kl, recon, kl


def ae_loss(x, x_hat):
    return ((x_hat - x) ** 2).sum(dim=(1, 2, 3)).mean()
